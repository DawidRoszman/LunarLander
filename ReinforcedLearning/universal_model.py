import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


env = gymnasium.make("LunarLander-v2", render_mode="rgb_array", enable_wind=False, gravity=-1)
for i in range(11):
    env = gymnasium.make("LunarLander-v2", render_mode="rgb_array", enable_wind=False, gravity=(-i-1))
    model = PPO("MlpPolicy", env, verbose=1)
    model.load("PPO_lunar_all")
    model.learn(total_timesteps=100_000)

    model.save("PPO_lunar_all")
    env.close()
