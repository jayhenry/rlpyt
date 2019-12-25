import numpy as np
import copy
import gym
from gym.spaces import Discrete, MultiDiscrete, Box

class SnakeEnv(gym.Env):
    SIZE=100
  
    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        # np.random.seed(123)
        # self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        # np.random.seed(None)
        # self.observation_space=Discrete(self.SIZE+1)
        self.observation_space=Box(0,2, shape=(self.SIZE*2,), dtype=np.uint8)
        self.action_space=Discrete(len(dices))

        self.ladders = {10: 90, 45: 91}
        self.ladders_in = np.zeros(self.SIZE)
        for k, v in self.ladders.items():
            self.ladders_in[k-1] = 1
            self.ladders_in[v-1] = 1
        print('Before ladders info:')
        print(self.ladders)
        tmp_ladders = copy.deepcopy(self.ladders)
        for k,v in tmp_ladders.items():
            self.ladders[v] = k
        print('ladders info:')
        print(self.ladders)
        print('dice ranges:')
        print(self.dices)
        self.pos = 1

    def reset(self):
        self.pos = 1
        ob = np.zeros(self.SIZE)
        ob[self.pos - 1] = 1
        ob = np.concatenate([ob, self.ladders_in])
        return ob

    def step(self, a):
        # state, reward, terminate, _
        step = np.random.randint(1, self.dices[a] + 1)
        self.pos += step
        if self.pos == 100:
            ob = np.zeros(self.SIZE)
            ob[self.pos - 1] = 1
            ob = np.concatenate([ob, self.ladders_in])
            return ob, 100, 1, {'traj_done': 1}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]

        ob = np.zeros(self.SIZE)
        ob[self.pos-1] = 1
        ob = np.concatenate([ob, self.ladders_in])
        return ob, -1, 0, {'traj_done': 0}

    def reward(self, s):
        if s == 100:
            return 100
        else:
            return -1

    def render(self):
        pass

if __name__ == '__main__':
    env = SnakeEnv(10, [3,6])
    env.reset()
    total_reward = 0
    while True:
        action = env.action_space.sample()
        state, reward, terminate, _ = env.step(action)
        total_reward += reward
        print(action, reward, state)
        if terminate == 1:
            break
    print("total_reward", total_reward)