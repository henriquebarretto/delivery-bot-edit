import pygame
import sys
import random
import heapq
import argparse
import csv
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Set, Tuple

# ==========================
# Utilitários
# ==========================
def manhattan(a: List[int], b: List[int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    def __init__(self, position: List[int]):
        self.position = position
        self.cargo = 0

    @abstractmethod
    def escolher_alvo(self, world: "World", current_steps: int,
                      avoid_positions: Optional[Set[Tuple[int, int]]] = None) -> Optional[List[int]]:
        raise NotImplementedError


class DefaultPlayer(BasePlayer):
    def get_remaining_steps(self, goal: Dict, current_steps: int) -> int:
        prioridade = goal["priority"]
        idade = current_steps - goal["created_at"]
        return prioridade - idade

    def escolher_alvo(self, world, current_steps, avoid_positions=None):
        avoid_positions = avoid_positions or set()
        sx, sy = self.position

        if self.cargo == 0 and world.packages:
            best = None
            best_dist = float('inf')
            for pkg in world.packages:
                if tuple(pkg) in avoid_positions:
                    continue
                d = manhattan(pkg, [sx, sy])
                if d < best_dist:
                    best_dist = d
                    best = pkg
            return best

        if world.goals:
            best_pos = None
            best_dist = float('inf')
            for goal in world.goals:
                if tuple(goal['pos']) in avoid_positions:
                    continue
                gx, gy = goal['pos']
                d = manhattan([gx, gy], [sx, sy])
                if d < best_dist:
                    best_dist = d
                    best_pos = goal['pos']
            return best_pos
        return None


class GreedyBestFirst(BasePlayer):
    def escolher_alvo(self, world, current_steps, avoid_positions=None):
        avoid_positions = avoid_positions or set()
        sx, sy = self.position
        candidates = []

        if self.cargo == 0 and world.packages:
            candidates = [pkg for pkg in world.packages if tuple(pkg) not in avoid_positions]
        elif world.goals:
            candidates = [goal['pos'] for goal in world.goals if tuple(goal['pos']) not in avoid_positions]

        if not candidates:
            return None

        return min(candidates, key=lambda c: manhattan(c, [sx, sy]))


class DeadlineAwarePlayer(BasePlayer):
    def __init__(self, position, urgent_threshold=10):
        super().__init__(position)
        self.urgent_threshold = urgent_threshold

    def escolher_alvo(self, world, current_steps, avoid_positions=None):
        """
        Estratégia:
        - Se tenho carga: entregar para a meta mais urgente (menor remaining), ponderando distância.
        - Se estou vazio: escolher pacote considerando tempo de entrega até alguma meta viável.
        - Evita coletar pacotes que certamente não podem ser entregues dentro do prazo.
        """
        avoid_positions = avoid_positions or set()
        sx, sy = self.position

        def remaining(goal: Dict) -> int:
            return goal['priority'] - (current_steps - goal['created_at'])

        # --- Se tenho carga: priorizar metas urgentes ---
        if self.cargo > 0 and world.goals:
            best_goal = None
            best_score = float('inf')
            for g in world.goals:
                if tuple(g['pos']) in avoid_positions:
                    continue
                rem = remaining(g)
                dist = manhattan(g['pos'], [sx, sy])
                score = (max(0, rem) * 2) + dist  # peso maior na urgência
                if score < best_score:
                    best_score = score
                    best_goal = g['pos']
            return best_goal

        # --- Se estou vazio: escolher pacote com melhor expectativa de entrega ---
        if self.cargo == 0 and world.packages:
            best_p, best_value = None, float('-inf')
            goals = [g for g in world.goals if tuple(g['pos']) not in avoid_positions]

            for p in world.packages:
                if tuple(p) in avoid_positions:
                    continue
                d_to_p = manhattan(p, [sx, sy])

                best_goal_value = float('-inf')
                for g in goals:
                    d_p_to_g = manhattan(g['pos'], p)
                    rem = remaining(g) - (d_to_p + d_p_to_g)
                    value = (50 if rem >= 0 else 0) - (d_to_p + d_p_to_g) - max(0, -rem)
                    if value > best_goal_value:
                        best_goal_value = value

                total_value = best_goal_value - d_to_p * 0.2
                if total_value > best_value:
                    best_value = total_value
                    best_p = p

            if best_p:
                return best_p

        # --- fallback: meta mais próxima/urgente ---
        if world.goals:
            return min(world.goals,
                       key=lambda g: (remaining(g), manhattan(g['pos'], [sx, sy])))['pos']
        return None

'''
edit:
Deadline agora calcula uma pontuação combinando urgência e distância (com peso maior para urgência),
o que evita que ele corra pra metas distantes com pouco ganho.
Quando sem carga, escolhe pacote pensando em qual meta pode ser salva ao pegá-lo
evita pegar pacotes que garantidamente falharão ao serem entregues.
'''

class SmartBatchPlayer(BasePlayer):
    def __init__(self, position, max_carry=2, urgent_threshold=8):
        super().__init__(position)
        self.max_carry = max_carry
        self.urgent_threshold = urgent_threshold

    def escolher_alvo(self, world, current_steps, avoid_positions=None):
        """
        Estratégia:
        - Entregar se houver metas urgentes viáveis.
        - Se ainda houver espaço de carga, decidir coletar pacotes considerando o valor esperado.
        - Usa max_carry de fato (coleta até encher antes de entregar, se fizer sentido).
        - Fallback: entregar mesmo metas atrasadas (ainda dá pontos líquidos).
        """
        avoid_positions = avoid_positions or set()
        sx, sy = self.position

        def remaining(g: Dict) -> int:
            return g['priority'] - (current_steps - g['created_at'])

        goals = [g for g in world.goals if tuple(g['pos']) not in avoid_positions]
        packages = [p for p in world.packages if tuple(p) not in avoid_positions]

        urgent_goals = [g for g in goals if remaining(g) <= self.urgent_threshold]

        # --- helper: valor líquido de pegar pacote e entregar ---
        def value_pick_and_deliver(p):
            d1 = manhattan(p, [sx, sy])
            best_val = float('-inf')
            for g in goals:
                d2 = manhattan(g['pos'], p)
                rem_after = remaining(g) - (d1 + d2)
                value = 50 - (d1 + d2) - max(0, -rem_after)
                if value > best_val:
                    best_val = value
            return best_val if goals else -d1

        # --- helper: valor líquido de entregar direto ---
        def value_deliver_direct(g):
            d = manhattan(g['pos'], [sx, sy])
            rem_after = remaining(g) - d
            return 50 - d - max(0, -rem_after)

        # --- Se houver metas urgentes ---
        if urgent_goals:
            if self.cargo > 0:
                return min(urgent_goals, key=lambda g: (manhattan(g['pos'], [sx, sy]), remaining(g)))['pos']
            if packages:
                # pegar pacote que mais ajuda salvar uma meta urgente
                return min(packages, key=lambda p: min(manhattan(p, ug['pos']) for ug in urgent_goals))

        # --- Se tenho carga e não há urgentes: avaliar entregar ---
        if self.cargo > 0 and goals:
            return max(goals, key=lambda g: value_deliver_direct(g))['pos']

        # --- Se ainda cabe carga: avaliar coletar ---
        if self.cargo < self.max_carry and packages:
            best_p = max(packages, key=lambda p: value_pick_and_deliver(p))
            return best_p

        # --- fallback: entregar mesmo atrasadas (ainda dá pontos líquidos) ---
        if goals and self.cargo > 0:
            return min(goals, key=lambda g: manhattan(g['pos'], [sx, sy]))['pos']

        # último recurso: andar até pacote mais próximo
        if packages:
            return min(packages, key=lambda p: manhattan(p, [sx, sy]))

        return None

'''
edit:
Em vez de regras fixas, o Smart estima o valor líquido de cada ação (50 pts por entrega menos custo em passos e penalidades potenciais).
Prioriza salvar metas urgentes, evitando pegar pacotes que não podem ser entregues a tempo.
Considera max_carry quando decide pegar mais pacotes.
Evita ações muito ruins (valor muito negativo).
'''

# ==========================
# CLASSE WORLD
# ==========================
class World:
    def __init__(self, seed=None, agent_name="default", agent_kwargs=None):
        # RNG para mapa fixo
        if seed is not None:
            random.seed(seed)

        # RNG separado para spawns dinâmicos (não é resetado pela seed global)
        import time
        self.rng = random.Random(time.time())

        agent_kwargs = agent_kwargs or {}

        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()

        self.walls = [(c, r) for r in range(self.maze_size) for c in range(self.maze_size) if self.map[r][c] == 1]

        self.total_items = 6
        self.packages = []
        while len(self.packages) < self.total_items + 1:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        self.goals = []
        self.player = self.generate_player(agent_name, agent_kwargs)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Delivery Bot - Agent: {agent_name}")

        self.package_image = None
        self.goal_image = None
        try:
            self.package_image = pygame.image.load("images/cargo.png")
            self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))
        except Exception:
            pass
        try:
            self.goal_image = pygame.image.load("images/operator.png")
            self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))
        except Exception:
            pass

        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)
        self.package_color = (255, 180, 0)
        self.goal_color = (0, 200, 100)

    def generate_obstacles(self):
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            for col in range(start, start + random.randint(5, 10)):
                if random.random() < 0.7 and 0 <= col < self.maze_size:
                    self.map[row][col] = 1
        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            for row in range(start, start + random.randint(5, 10)):
                if random.random() < 0.7 and 0 <= row < self.maze_size:
                    self.map[row][col] = 1
        block_size = random.choice([4, 6])
        top_row = random.randint(0, self.maze_size - block_size)
        top_col = random.randint(0, self.maze_size - block_size)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self, agent_name, agent_kwargs):
        while True:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                if agent_name == 'default':
                    return DefaultPlayer([x, y])
                elif agent_name == 'greedy':
                    return GreedyBestFirst([x, y])
                elif agent_name == 'deadline':
                    return DeadlineAwarePlayer([x, y], **agent_kwargs)
                elif agent_name == 'smart':
                    return SmartBatchPlayer([x, y], **agent_kwargs)
                else:
                    raise ValueError("Agente desconhecido")


    def random_free_cell(self) -> List[int]:
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            occupied = (
                self.map[y][x] == 1 or
                [x, y] in self.packages or
                [x, y] == self.player.position or
                any(g['pos'] == [x, y] for g in self.goals)
            )
            if not occupied:
                return [x, y]

    def add_goal(self, created_at_step: int):
        pos = self.random_free_cell()
        priority = self.rng.randint(40, 110)
        self.goals.append({"pos": pos, "priority": priority, "created_at": created_at_step})

    def can_move_to(self, pos: List[int]) -> bool:
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.map[y][x] == 0
        return False

    def draw_world(self, path: Optional[List[List[int]]] = None):
        self.screen.fill(self.ground_color)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)

        for pkg in self.packages:
            x, y = pkg
            if self.package_image:
                self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
            else:
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, self.package_color, rect)

        for goal in self.goals:
            x, y = goal['pos']
            if self.goal_image:
                self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
            else:
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, self.goal_color, rect)

        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(x * self.block_size + self.block_size // 4,
                                   y * self.block_size + self.block_size // 4,
                                   self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)

        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()

# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed: Optional[int] = None, agent: str = 'smart', delay_ms: int = 60,
                 agent_kwargs: Optional[dict] = None, sticky_target: bool = True):
        self.world = World(seed, agent, agent_kwargs)
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = delay_ms
        self.path: List[List[int]] = []
        self.num_deliveries = 0
        self.sticky_target = sticky_target
        self.agent_name = agent
        self.seed = str(seed)
        self.max_carry = agent_kwargs.get("max_carry") if agent_kwargs else None
        self.urgent_threshold = agent_kwargs.get("urgent_threshold") if agent_kwargs else None

        # Spawn inicial de metas: 1 meta no passo 0 (alinhado com o código do professor)
        self.world.add_goal(created_at_step=0)
        self.goal_spawns = 1

        # Agenda de novos spawns para totalizar 6 metas
        self.spawn_intervals = (
            [self.world.rng.randint(2, 5)] +
            [self.world.rng.randint(5, 10)] +
            [self.world.rng.randint(10, 15) for _ in range(3)]
        )

        self.next_spawn_step = self.spawn_intervals.pop(0)

        # Alvo corrente
        self.current_target: Optional[List[int]] = None

        # Alvos temporariamente evitados (atingíveis? se A* falhar para eles)
        # mapeia (x,y) -> expire_step
        self.avoid_targets: Dict[Tuple[int, int], int] = {}

        # LOG HISTORY
        self.history = [] #Lista pra guardar o desempenho 
        # salva a data/hora de início da execução Y-M-D // H:M:S
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_state() # registro do estado inicial antes do jogo começar
    
    # Log history - csv
    def log_state(self):
        delayed_count = sum(1 for g in self.world.goals if (self.steps - g['created_at']) > g['priority'])
        self.history.append({
            "datetime": self.run_timestamp,
            "agent": self.agent_name,
            "seed": self.seed,
            "steps": self.steps,
            "score": self.score,
            "cargo": self.world.player.cargo,
            "deliveries": self.num_deliveries,
            "active_goals": len(self.world.goals),
            "delayed_goals": delayed_count,
            "sticky_target": self.sticky_target,
            "max_carry": self.max_carry,
            "urgent_threshold": self.urgent_threshold
    })

    # Salvamento do history
    def save_results(self, filename: str):
        if not self.history:
            print("Nenhum dado para salvar.")
            return
        # pega só o último registro (desempenho final)
        final_result = self.history[-1]
        # se o arquivo ainda não existe, cria com cabeçalho
        import os
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=final_result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(final_result)
        print(f"Resultado final salvo em {filename}")

    # A*
    def heuristic(self, a: List[int], b: List[int]) -> int:
        return manhattan(a, b)

    def astar(self, start: List[int], goal: List[int]) -> List[List[int]]:
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))

        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                data: List[List[int]] = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                data.reverse()
                return data

            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1

                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue

                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(list(neighbor), goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return []

    # Spawns & penalidades
    def maybe_spawn_goal(self):
        while (self.next_spawn_step is not None and
               self.steps >= self.next_spawn_step and
               self.goal_spawns < self.world.total_items):
            self.world.add_goal(created_at_step=self.steps)
            self.goal_spawns += 1
            if self.spawn_intervals and self.goal_spawns < self.world.total_items:
                self.next_spawn_step += self.spawn_intervals.pop(0)
            else:
                self.next_spawn_step = None

    def delayed_goals_penalty(self) -> int:
        delayed = 0
        for g in self.world.goals:
            age = self.steps - g['created_at']
            if age > g['priority']:
                delayed += 1
        return delayed

    def get_goal_at(self, pos: List[int]) -> Optional[Dict]:
        for g in self.world.goals:
            if g['pos'] == pos:
                return g
        return None

    def idle_tick(self):
        self.steps += 1
        self.score -= 1
        self.score -= self.delayed_goals_penalty()
        self.maybe_spawn_goal()
        self.world.draw_world(self.path)
        pygame.time.wait(self.delay)

    # Loop principal
    def game_loop(self):
        while self.running:
            # condição de término
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            # permite spawns
            self.maybe_spawn_goal()

            # limpa avoid_targets expirados
            expired = [p for p, exp in self.avoid_targets.items() if exp <= self.steps]
            for p in expired:
                del self.avoid_targets[p]

            current_avoid = set(self.avoid_targets.keys())

            # escolhe alvo quando necessário
            if not self.sticky_target or self.current_target is None:
                target = self.world.player.escolher_alvo(self.world, self.steps, current_avoid)
                if target is None:
                    self.idle_tick()
                    continue
                self.current_target = target

            # Se já estamos no alvo, processa imediatamente (sem chamar A*)
            if self.world.player.position == self.current_target:
                self._process_arrival()
                self.current_target = None
                # log e continue
                delayed_count = sum(1 for g in self.world.goals if (self.steps - g['created_at']) > g['priority'])
                print(f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, "
                      f"Entregas: {self.num_deliveries}, Goals ativos: {len(self.world.goals)}, Atrasados: {delayed_count}")
                continue

            # planeja caminho
            self.path = self.astar(self.world.player.position, self.current_target)
            if not self.path:
                print("Nenhum caminho encontrado para o alvo", self.current_target)
                # marca temporariamente como evitável para não ficar em loop infinito
                self.avoid_targets[(self.current_target[0], self.current_target[1])] = self.steps + 50
                self.current_target = None
                continue

            # segue caminho
            for pos in self.path:
                self.world.player.position = pos
                self.steps += 1

                # custos e penalidades
                self.score -= 1
                self.score -= self.delayed_goals_penalty()

                # spawns
                self.maybe_spawn_goal()

                # desenha
                self.world.draw_world(self.path)
                pygame.time.wait(self.delay)

                # log do estado a cada passo
                self.log_state()

                # eventos
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                if not self.running:
                    break

                # se não for sticky, reavalie já
                if not self.sticky_target:
                    break

            if not self.running:
                break

            # se não for sticky, volte ao loop principal pra reavaliar
            if not self.sticky_target:
                self.current_target = None
                continue

            # ao chegar no fim do path, processa chegada
            if self.world.player.position == self.current_target:
                self._process_arrival()

            # reset do alvo
            self.current_target = None

            # log
            delayed_count = sum(1 for g in self.world.goals if (self.steps - g['created_at']) > g['priority'])
            print(f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, "
                  f"Entregas: {self.num_deliveries}, Goals ativos: {len(self.world.goals)}, Atrasados: {delayed_count}")

        print("Fim de jogo!")
        print("Total de passos:", self.steps)
        print("Pontuação final:", self.score)

        self.log_state()
        self.save_results("resultados.csv")

        pygame.quit()
            
    def _process_arrival(self):
        # Chamada quando jogador está exatamente em current_target
        pos = self.world.player.position

        # coleta?
        if pos in self.world.packages:
            self.world.player.cargo += 1
            self.world.packages.remove(pos)
            print(f"[COLETA] Pacote coletado em {pos} | Cargo: {self.world.player.cargo}")
            return

        # entrega?
        goal = self.get_goal_at(pos)
        if goal is not None:
            if self.world.player.cargo > 0:
                self.world.player.cargo -= 1
                self.num_deliveries += 1
                self.world.goals.remove(goal)
                self.score += 50
                print(f"[ENTREGA] Pacote entregue em {pos} | Cargo: {self.world.player.cargo} | "
                      f"Priority: {goal['priority']} | Age: {self.steps - goal['created_at']}")
            else:
                # chegou em meta sem carga -> nada a fazer
                print(f"[INFO] Chegou em meta {pos} sem carga. Nada feito.")

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================

def build_agent_kwargs(agent: str, max_carry: int, urgent_threshold: int) -> dict:
    if agent == 'deadline':
        return {'urgent_threshold': urgent_threshold}
    if agent == 'smart':
        return {'max_carry': max_carry, 'urgent_threshold': urgent_threshold}
    return {}

from menu import Menu

# MAIN
if __name__ == "__main__":
    menu = Menu()
    config = menu.loop()  # usuário escolhe no menu

    agent = config["agent"]
    max_carry = config["max_carry"]
    urgent_threshold = config["urgent_threshold"]

    # monta kwargs do agente de acordo com o tipo
    agent_kwargs = build_agent_kwargs(agent, max_carry, urgent_threshold)

    print(f"[DEBUG] Agent: {agent}, max_carry={max_carry}, urgent_threshold={urgent_threshold}")

    maze = Maze(seed=config["seed"], agent=agent,
                delay_ms=60, agent_kwargs=agent_kwargs,
                sticky_target=config["sticky_target"])
    maze.game_loop()
