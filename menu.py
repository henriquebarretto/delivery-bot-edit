import pygame
import sys

class Menu:
    def __init__(self):
        pygame.init()
        self.width, self.height = 700, 500
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot - Configurações")

        self.font = pygame.font.SysFont("arial", 28)
        self.small_font = pygame.font.SysFont("arial", 22)

        # Opções padrão
        self.options = {
            "agent": "smart",
            "seed": None,
            "sticky_target": False,
            "max_carry": 2,
            "urgent_threshold": 10
        }

        self.agents = ["default", "greedy", "deadline", "smart"]
        self.agent_dropdown_open = False

        self.input_active = None
        self.inputs = {
            "seed": "",
            "max_carry": "2",
            "urgent_threshold": "10"
        }

        self.running = True
        self.result = None

    def draw_text(self, text, x, y, color=(0, 0, 0), center=False, font=None):
        font = font or self.font
        label = font.render(text, True, color)
        rect = label.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(label, rect)

    def draw_input(self, key, x, y, w=180, h=40):
        rect = pygame.Rect(x, y, w, h)
        color = (200, 200, 250) if self.input_active == key else (240, 240, 240)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 150), rect, 2, border_radius=8)

        text_surface = self.small_font.render(self.inputs[key], True, (0, 0, 0))
        self.screen.blit(text_surface, (rect.x + 8, rect.y + 8))

        return rect

    def draw_button(self, text, rect, color, hover_color, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if rect.collidepoint(mouse):
            pygame.draw.rect(self.screen, hover_color, rect, border_radius=12)
            if click[0] == 1 and action:
                action()
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=12)

        label = self.font.render(text, True, (255, 255, 255))
        label_rect = label.get_rect(center=rect.center)
        self.screen.blit(label, label_rect)

    def draw_dropdown(self, x, y, w=200, h=40):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (240, 240, 240), rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 150), rect, 2, border_radius=8)

        self.draw_text(self.options["agent"] + " ▼", rect.x + 10, rect.y + 7, font=self.small_font)

        if self.agent_dropdown_open:
            for i, agent in enumerate(self.agents):
                opt_rect = pygame.Rect(x, y + (i+1)*h, w, h)
                color = (200, 220, 250) if opt_rect.collidepoint(pygame.mouse.get_pos()) else (240, 240, 240)
                pygame.draw.rect(self.screen, color, opt_rect, border_radius=6)
                pygame.draw.rect(self.screen, (100, 100, 150), opt_rect, 1, border_radius=6)
                self.draw_text(agent, opt_rect.x + 10, opt_rect.y + 7, font=self.small_font)

                if opt_rect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
                    self.options["agent"] = agent
                    self.agent_dropdown_open = False

        return rect

    def draw_checkbox(self, x, y):
        rect = pygame.Rect(x, y, 25, 25)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=4)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 2, border_radius=4)
        if self.options["sticky_target"]:
            pygame.draw.line(self.screen, (0, 120, 250), (x+5, y+12), (x+12, y+20), 3)
            pygame.draw.line(self.screen, (0, 120, 250), (x+12, y+20), (x+20, y+5), 3)
        return rect

    def set_default(self):
        self.options = {
            "agent": "smart",
            "seed": None,
            "sticky_target": False,
            "max_carry": 2,
            "urgent_threshold": 10
        }
        self.inputs = {
            "seed": "",
            "max_carry": "2",
            "urgent_threshold": "10"
        }

    def start_game(self):
        self.options["seed"] = None if self.inputs["seed"] == "" else int(self.inputs["seed"])
        self.options["max_carry"] = int(self.inputs["max_carry"])
        self.options["urgent_threshold"] = int(self.inputs["urgent_threshold"])
        self.result = self.options
        self.running = False

    def toggle_sticky(self):
        self.options["sticky_target"] = not self.options["sticky_target"]

    def loop(self):
        while self.running:
            self.screen.fill((245, 245, 255))

            # Título
            self.draw_text("Delivery Bot - Configurações", self.width//2, 40, color=(0, 60, 160), center=True)

            # Inputs
            self.draw_text("Seed:", 80, 170)
            seed_rect = self.draw_input("seed", 180, 165)

            self.draw_text("Max Carry:", 80, 230)
            max_rect = self.draw_input("max_carry", 230, 225)

            self.draw_text("Urgent Threshold:", 80, 290)
            urgent_rect = self.draw_input("urgent_threshold", 330, 285)

            # Checkbox sticky
            sticky_rect = self.draw_checkbox(80, 350)
            self.draw_text("Sticky Target", 120, 350)
            
            # Dropdown Agente
            self.draw_text("Agente:", 80, 100)
            agent_rect = self.draw_dropdown(200, 95)

            # Botões
            self.draw_button("Default", pygame.Rect(100, 420, 160, 50), (0, 180, 0), (0, 220, 0), self.set_default)
            self.draw_button("Iniciar", pygame.Rect(340, 420, 160, 50), (0, 0, 180), (0, 0, 220), self.start_game)

            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if agent_rect.collidepoint(event.pos):
                        self.agent_dropdown_open = not self.agent_dropdown_open
                    elif seed_rect.collidepoint(event.pos):
                        self.input_active = "seed"
                    elif max_rect.collidepoint(event.pos):
                        self.input_active = "max_carry"
                    elif urgent_rect.collidepoint(event.pos):
                        self.input_active = "urgent_threshold"
                    elif sticky_rect.collidepoint(event.pos):
                        self.toggle_sticky()
                    else:
                        self.input_active = None

                if event.type == pygame.KEYDOWN and self.input_active:
                    if event.key == pygame.K_BACKSPACE:
                        self.inputs[self.input_active] = self.inputs[self.input_active][:-1]
                    elif event.unicode.isdigit():
                        self.inputs[self.input_active] += event.unicode

            pygame.display.flip()
            pygame.time.wait(100)

        return self.result
