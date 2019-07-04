from gym.envs.registration import register

""" Register custom environments at OpenAI Gym
"""

register(
    id='Sawtooth-v0',
    entry_point='tensorbox.envs.sawtooth_env:SawtoothWaveEnv',
    max_episode_steps=512,
)

