import gymnasium
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


env = gymnasium.make("LunarLander-v2", render_mode="human")

# model = A2C("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=800_000)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
model.save("A2C_lunar")
