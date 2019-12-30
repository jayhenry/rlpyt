#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
 Author: zhaopenghao
 Create Time: 2019/12/25 下午4:29
"""
import gym


# env_name = 'CartPole-v0'
# env_name = 'LunarLander-v2'
# env_name = 'Taxi-v2'
# env_name = 'Blackjack-v0'
# env_name = 'Copy-v0'
env_name = 'MountainCar-v0'
# env_name = 'Acrobot-v1'

# Continuous
# env_name = 'InvertedPendulum-v2'  # install mujoco_py, https://github.com/openai/mujoco-py/
# env_name = 'LunarLanderContinuous-v2'
# env_name = 'BipedalWalker-v2'
# env_name = 'BipedalWalkerHardcore-v2'
# env_name = 'CarRacing-v0'
# env_name = 'Pendulum-v0'
print("env_name:", env_name)
env = gym.make(env_name)

print("env.spec.max_episode_steps", env.spec.max_episode_steps)
print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = None
try:
    if isinstance(env.observation_space, gym.spaces.Discrete):
        ob_dim = env.observation_space.n
    else:
        ob_dim = env.observation_space.shape[0]
except Exception as e:
    print("Get ob_dim error:", e)

ac_dim = None
try:
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
except Exception as e:
    print("Get ac_dim error:", e)
print("discrete", discrete, "ob dim", ob_dim, "ac_dim", ac_dim)

env.seed(1234)
ob = env.reset()
print("reset1 ob", ob)
env.seed(1234)
ob = env.reset()
print("reset2 ob", ob)
env.seed(None)

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
        print("action {}; reward {}; state {}".format(action, reward, state))
        if terminate == 1:
            break
    print("total_reward", total_reward)

    env.close()