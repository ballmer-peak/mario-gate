import gym
import random
import gym_pull

env = gym.make("ppaquette/SuperMarioBros-1-1-Tiles-v0")
#env = gym.make("SuperMarioBros-1-1-Tiles-v0")
observation = env.reset()

print(observation)


while(1):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(observation)



