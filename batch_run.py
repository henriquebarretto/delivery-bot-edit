from main import Maze, build_agent_kwargs

# Configurações do benchmark
seeds = [5, 10, 15, 32]

agents = ["default", "greedy", "deadline", "smart"]

# combinações pro agent smart
smart_settings = [
    (1, 10),  # max_carry=1, urgent_threshold=10
    (2, 10),  # max_carry=2, urgent_threshold=10
    (1, 15),  # max_carry=1, urgent_threshold=15
    (2, 15),  # max_carry=2, urgent_threshold=15
]
# combinações pro agente deadline
deadline_settings=[5, 10, 15, 20]

sticky_options = [False, True]

# Loop principal
for seed in seeds:
    for sticky in sticky_options:
        for agent in agents:
            if agent == "smart":
                # Smart -> precisa de max_carry e urgent_threshold
                for max_carry, urgent_threshold in smart_settings:
                    print(f"\n>>> seed={seed}, agent={agent}, carry={max_carry}, urg={urgent_threshold}, sticky={sticky}")
                    agent_kwargs = build_agent_kwargs(agent, max_carry=max_carry, urgent_threshold=urgent_threshold)
                    maze = Maze(seed=seed, agent=agent, delay_ms=0,
                                agent_kwargs=agent_kwargs, sticky_target=sticky)
                    maze.game_loop()

            elif agent == "deadline":
                # Deadline -> só precisa de urgent_threshold
                for urgent_threshold in deadline_settings:
                    print(f"\n>>> seed={seed}, agent={agent}, urg={urgent_threshold}, sticky={sticky}")
                    agent_kwargs = build_agent_kwargs(agent, max_carry=None, urgent_threshold=urgent_threshold)
                    maze = Maze(seed=seed, agent=agent, delay_ms=0,
                                agent_kwargs=agent_kwargs, sticky_target=sticky)
                    maze.game_loop()

            else:
                # Default e Greedy -> sem parâmetros extras
                print(f"\n>>> seed={seed}, agent={agent}, sticky={sticky}")
                agent_kwargs = build_agent_kwargs(agent, max_carry=None, urgent_threshold=None)
                maze = Maze(seed=seed, agent=agent, delay_ms=0,
                            agent_kwargs=agent_kwargs, sticky_target=sticky)
                maze.game_loop()

