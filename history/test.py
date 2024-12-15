import gymnasium as gym
import keyboard
import time


env = gym.make('Boxing-v0', render_mode='human', full_action_space=True)

env.reset()

number_of_actions = env.action_space.n

meaning = env.unwrapped.get_action_meanings()
