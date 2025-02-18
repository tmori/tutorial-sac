import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HelloWorldPIDEnv(gym.Env):
    """SAC のテスト用カスタム環境（PID 最適化の基本）"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        #print("HelloWorldPIDEnv initialized")
        # 期待する PID パラメータ（目標値）
        self.target_pid = np.array([0.1, 0.5, 1.0], dtype=np.float32)

        # アクション空間（SAC が決定する PID パラメータ: Kp, Ki, Kd）
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32)

        # 状態空間（ダミー: SAC の学習には使わない）
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        #print("HelloWorldPIDEnv reset")
        """環境をリセット（毎回 `target_pid` は変えず、学習のために初期 `obs` を設定）"""
        super().reset(seed=seed)
        self.last_action = np.random.uniform(0.0, 2.0, size=(3,))  # ランダムな初期 PID 値
        obs = self.last_action - self.target_pid  # 誤差を観測値にする
        return np.array(obs, dtype=np.float32), {}


    def step(self, action):
        #print("HelloWorldPIDEnv step")
        """SAC のエージェントが決定した PID パラメータを評価"""
        reward = -np.linalg.norm(action - self.target_pid)  # 目標との差を報酬にする

        self.last_action = action  # 現在の `action` を保存

        obs = action - self.target_pid  # 誤差を観測値にする
        done = True  # 1 ステップで終了（変更の余地あり）

        return np.array(obs, dtype=np.float32), reward, done, False, {}


    def render(self):
        #print("HelloWorldPIDEnv render")
        """（今回はレンダリング不要）"""
        pass

    def close(self):
        #print("HelloWorldPIDEnv close")
        """終了処理"""
        pass
