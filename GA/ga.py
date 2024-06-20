import gymnasium as gym
import pygad
import pickle

# Define the Lunar Lander environment
env = gym.make("LunarLander-v2")

# Define action space (0: do nothing, 1: left orientation engine, 2: main engine, 3: right orientation engine)
ACTIONS = [0, 1, 2, 3]
ACTION_SPACE = len(ACTIONS)

# Maximum number of moves allowed
MAX_MOVES_ALLOWED = 200


# Define the fitness function
def fitness_func(model, solution, solution_idx):
    observation = env.reset()
    total_reward = 0
    for step in solution:
        step = int(step)  # Ensure the step is an integer
        observation, reward, terminated, truncated, info = env.step(step)

        # Punish for too fast velocity
        if observation[3] > 0.4:
            total_reward -= 50

        # Punish for too much tilt
        if abs(observation[4]) > 0.5:
            total_reward -= 30

        total_reward += float(reward)
        if terminated or truncated:
            break
    return total_reward


# Number of chromosomes in a population
sol_per_pop = 100
# Number of genes in the chromosome (number of moves)
num_genes = MAX_MOVES_ALLOWED
# Number of parents to participate in further mating (about 50%)
num_parents_mating = 50
# Number of generations
num_generations = 300
# Number of parents to keep (a few percent)
keep_parents = 10

# Parent selection type
parent_selection_type = "sss"

# What type of chromosome crossover (single-point / two-point etc.)
crossover_type = "single_point"

# Percentage of genes affected by mutation
mutation_type = "swap"
mutation_percent_genes = "default"

# Initializing the genetic algorithm
ga_instance = pygad.GA(
    gene_space=ACTIONS,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
)

# Running the genetic algorithm
ga_instance.run()

# Displaying the summary of the best solution found
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution: {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")

# Save the solution to a file
with open("best_solution_lunar_lander.pkl", "wb") as f:
    pickle.dump(solution, f)

# Plotting the fitness over generations
ga_instance.plot_fitness(save_dir="fitness_plot_lunar_lander")

# Visualizing the best solution
env = gym.make(
    "LunarLander-v2", render_mode="human"
)  # Enable human mode for visualization
env.reset()
env.render()

for step in solution:
    observation, reward, terminated, truncated, info = env.step(
        int(step)
    )  # Ensure the step is an integer
    env.render()

    if terminated or truncated:
        break  # Stop rendering if episode terminates

env.close()
