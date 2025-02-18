import gymnasium as gym
from stable_baselines3 import SAC

# カスタム環境を登録（初回のみ）
import register_pid_env  # これで `HelloWorldPID-v0` が使えるようになる

# 環境を作成
print("Creating environment...")
env = gym.make("HelloWorldPID-v0")

# SAC モデルを作成
print("Creating model...")
model = SAC("MlpPolicy", env, verbose=1, gradient_steps=1)

# 学習
print("Training...")
model.learn(total_timesteps=10000)

# 学習済みモデルを保存
print("Saving model...")
model.save("sac_pid")

# 学習したモデルでテスト
print("Testing...")
env = gym.make("HelloWorldPID-v0")
obs, info = env.reset()

print("Start testing...")
for _ in range(10):  # 10 回テスト
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

env.close()
