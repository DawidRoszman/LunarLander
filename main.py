import gymnasium
import numpy as np


def update_action_probabilities(chance_for_action, reward, alpha=0.1, epsilon=0.1):
    """
    Updates action probabilities based on rewards with Epsilon-Greedy exploration.

    Args:
        chance_for_action (list): List of current action probabilities.
        reward (float): Reward received from the environment.
        alpha (float, optional): Learning rate for updating probabilities. Defaults to 0.1.
        epsilon (float, optional): Exploration rate (probability of choosing a random action). Defaults to 0.1.

    Returns:
        list: Updated list of action probabilities.
    """
    if np.random.rand() < epsilon:
        return chance_for_action  # Don't update, choose random action for exploration

    updated_chance_for_action = np.zeros_like(chance_for_action)
    total_adjustment = 0

    for i, action_probability in enumerate(chance_for_action):
        adjustment = 0
        if i == np.argmax(chance_for_action):
            if reward > 0:  # Increase probability for good action
                adjustment = alpha * reward
            else:  # Decrease probability for bad action (slightly increase others)
                adjustment = -alpha * reward
                total_adjustment += adjustment
        updated_chance_for_action[i] = action_probability + adjustment

    # Normalize probabilities after adjustments
    updated_chance_for_action += total_adjustment / (len(chance_for_action) - 1)
    updated_chance_for_action = np.clip(updated_chance_for_action, 0, 1)
    updated_chance_for_action /= np.sum(updated_chance_for_action)
    return updated_chance_for_action


chance_for_action = [0.22, 0.27, 0.26, 0.25]
env = gymnasium.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
learning_rate = 0.1
epsilon = 0.85
total_reward = 0
previous_total_reward = 0
previous_action_probabilites = None
while True:
    action = np.random.choice([0, 1, 2, 3], p=chance_for_action)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)

    # Update action probabilities based on reward
    chance_for_action = update_action_probabilities(
        chance_for_action, reward, learning_rate, epsilon
    )

    print("Reward:", reward)
    print("Action probabilities:", chance_for_action)
    print("Total reward:", total_reward)

    if terminated or truncated:
        print("Total reward:", total_reward)
        if total_reward > 200:
            print("Lunar Lander successfully landed!")
            break
        if (
            total_reward < previous_total_reward
            and previous_action_probabilites is not None
        ):
            chance_for_action = previous_action_probabilites
        else:
            previous_action_probabilites = chance_for_action
        previous_total_reward = total_reward
        total_reward = 0
        epsilon = max(epsilon - 0.01, 0.1)
        observation, info = env.reset()

env.close()
