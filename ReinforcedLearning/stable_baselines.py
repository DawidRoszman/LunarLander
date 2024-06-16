import gymnasium
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


env = gymnasium.make("LunarLander-v2", render_mode="rgb_array", enable_wind=True, gravity=-11, wind_power=30)

# model = A2C("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)
# model.load("PPO_lunar_all")
model.learn(total_timesteps=1_000_000)

# evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
model.save("PPO_lunar_hard")
