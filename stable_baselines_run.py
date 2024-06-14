from stable_baselines3 import PPO, A2C
import gymnasium

env = gymnasium.make("LunarLander-v2", render_mode="human")
# model = PPO.load("PPO_lunar", env=env)
model = A2C.load("A2C_lunar", env=env)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
