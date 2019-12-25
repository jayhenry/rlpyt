#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
 Author: zhaopenghao
 Create Time: 2019/12/25 下午4:29
"""
import gym


env_name = 'CartPole-v0'
env = gym.make(env_name)

print("env.spec.max_episode_steps", env.spec.max_episode_steps)
print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
print("discrete", discrete, "ob dim", ob_dim, "ac_dim", ac_dim)

#todo: sample and render
if __name__ == '__main__':
    import time
    ob = env.reset()
    print("reset ob", ob)
    total_reward = 0
    while True:
        env.render()
        time.sleep(0.1)
        action = env.action_space.sample()
        state, reward, terminate, _ = env.step(action)
        total_reward += reward
        print(action, reward, state)
        if terminate == 1:
            break
    print("total_reward", total_reward)

    env.close()