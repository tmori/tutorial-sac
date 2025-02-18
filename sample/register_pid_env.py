from gymnasium.envs.registration import register

register(
    id="HelloWorldPID-v0",
    entry_point="pid_env:HelloWorldPIDEnv",
    max_episode_steps=1,  # 1 ステップ = 1 試行
)
