import pickle
import gymnasium as gym

# Load solution from file
with open("best_solution_lunar_lander.pkl", "rb") as f:
    solution = pickle.load(file=f)

# Visualizing the best solution
env = gym.make(
    "LunarLander-v2", render_mode="human"
)  # Enable human mode for visualization
env.reset()
env.render()

total_reward = 0
while True:
    for step in solution:
        observation, reward, terminated, truncated, info = env.step(int(step))
        total_reward += float(reward)
        env.render()

        if terminated or truncated:
            break
    if total_reward > 200:
        print("Lunar Lander successfully landed!")
        break
    observation, info = env.reset()
    total_reward = 0

env.close()
