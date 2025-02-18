import gymnasium as gym
from stable_baselines3 import SAC

# 連続アクション環境（SAC に対応）
env = gym.make("LunarLanderContinuous-v3")

# SAC モデルを作成
model = SAC("MlpPolicy", env, verbose=1)

# 学習（試しに10,000ステップ）
model.learn(total_timesteps=10000)

# 学習済みモデルを保存
model.save("sac_lunarlander")

# 学習したモデルを使ってテスト
env = gym.make("LunarLanderContinuous-v3", render_mode="human")
obs, info = env.reset()

for _ in range(1000):  # 1000ステップ分実行
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()
print("SAC による強化学習が完了しました！")
