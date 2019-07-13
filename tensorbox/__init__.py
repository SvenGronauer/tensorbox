from gym.envs.registration import register

""" Register custom environments at OpenAI Gym
"""

register(
    id='Sawtooth-v0',
    entry_point='tensorbox.envs.sawtooth_env:SawtoothWaveEnv',
    max_episode_steps=512,
)

register(
    id='PT1System-v0',
    entry_point='tensorbox.envs.control_systems:PT1SystemEnv'
)

register(
    id='PT2System-v0',
    entry_point='tensorbox.envs.control_systems:PT2SystemEnv'
)
