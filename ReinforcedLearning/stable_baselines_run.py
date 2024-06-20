from stable_baselines3 import PPO, A2C
import gymnasium

env = gymnasium.make(
    "LunarLander-v2", render_mode="human", enable_wind=True, gravity=-10, wind_power=10
)
# model = PPO.load("PPO_lunar", env=env)
model = PPO.load("PPO_lunar", env=env)
vec_env = model.get_env()
obs = vec_env.reset()
total_reward = 0
while 1:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    total_reward += reward
    print(f"Reward: {reward}, Total Reward: {total_reward}")
    if done:
        if total_reward > 200:
            print(f"Success! Reward: {total_reward}")
            break
        else:
            total_reward = 0
    vec_env.render("human")
