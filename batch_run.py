from main import Maze, build_agent_kwargs
import pandas as pd

# Configurações do benchmark
seeds = [5, 15, 150]
agents = ["default", "greedy", "deadline", "smart"]

# combinações pro agent smart
smart_settings = [
    (1, 10),
    (2, 10),
    (1, 15),
    (2, 15),
]

# combinações pro deadline
deadline_settings = [10, 15]

sticky_options = [False, True]

# número de repetições para cada configuração
num_runs = 3

# Arquivo de saída com resultados brutos
results_file = "resultados.csv"

# Loop principal
for seed in seeds:
    for sticky in sticky_options:
        for agent in agents:
            if agent == "smart":
                for max_carry, urgent_threshold in smart_settings:
                    for run in range(num_runs):
                        print(f"\n>>> seed={seed}, agent={agent}, carry={max_carry}, urg={urgent_threshold}, "
                              f"sticky={sticky}, run={run+1}/{num_runs}")
                        agent_kwargs = build_agent_kwargs(agent, max_carry=max_carry, urgent_threshold=urgent_threshold)
                        maze = Maze(seed=seed, agent=agent, delay_ms=0,
                                    agent_kwargs=agent_kwargs, sticky_target=sticky)
                        maze.game_loop()

            elif agent == "deadline":
                for urgent_threshold in deadline_settings:
                    for run in range(num_runs):
                        print(f"\n>>> seed={seed}, agent={agent}, urg={urgent_threshold}, "
                              f"sticky={sticky}, run={run+1}/{num_runs}")
                        agent_kwargs = build_agent_kwargs(agent, max_carry=None, urgent_threshold=urgent_threshold)
                        maze = Maze(seed=seed, agent=agent, delay_ms=0,
                                    agent_kwargs=agent_kwargs, sticky_target=sticky)
                        maze.game_loop()

            else:
                for run in range(num_runs):
                    print(f"\n>>> seed={seed}, agent={agent}, sticky={sticky}, run={run+1}/{num_runs}")
                    agent_kwargs = build_agent_kwargs(agent, max_carry=None, urgent_threshold=None)
                    maze = Maze(seed=seed, agent=agent, delay_ms=0,
                                agent_kwargs=agent_kwargs, sticky_target=sticky)
                    maze.game_loop()