
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""
import random
import numpy as np

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.algos.pg.a2c import A2C
# from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

import gym
import torch
from torch import nn
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, to_onehot
import torch.nn.functional as F



class ModelCls(nn.Module):
    def __init__(self, ob_dim, ac_dim, **kwargs):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.model = MlpModel(self.ob_dim, [64, 64], output_size=None,
                              nonlinearity=torch.nn.Tanh,  # torch.nn.Tanh
                              )
        self.pi = torch.nn.Linear(64, self.ac_dim)
        self.model2 = MlpModel(self.ob_dim, [64, 64], output_size=None,
                               nonlinearity=torch.nn.ReLU,  # torch.nn.Tanh,
                               )
        self.value = torch.nn.Linear(64, 1)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        # observation = observation.squeeze(-1)
        # print("ob shape: {}".format(observation.shape))
        # ob2 = observation[1:]
        # ob1 = to_onehot(observation[0], 100)
        # print("ob1 shape: {}, ob2 shape {}".format(ob1.shape, ob2.shape))
        # observation = torch.cat([ob1, ob2], dim=-1)
        # print("After ob shape: {}".format(observation.shape))
        # print("ob onehot:", observation)
        input = observation.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, input_shape = infer_leading_dims(input, 1)

        fc_out = self.model(input.view(T * B, *input_shape))  # Fold if T dimension.
        pi = F.softmax(self.pi(fc_out), dim=-1)

        fc_out = self.model2(input.view(T * B, *input_shape))  # Fold if T dimension.
        v = self.value(fc_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


def build_and_train(env_id="Hopper-v3", run_ID=0, cuda_idx=None):
    seed = 1
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # 但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    # torch.cuda.manual_seed(seed)
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    batch_steps = 1e3
    log_steps = batch_steps
    n_steps = 30 * batch_steps
    eval_steps = batch_steps
    eval_trajs = 100
    lr = 5e-3
    animate = True
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=int(batch_steps),  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(eval_steps),
        eval_max_trajectories=eval_trajs,
        animate=animate,
    )

    # env = sampler.collector.envs[0]
    env = gym.make(env_id)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    print("discrete", discrete, "ob dim", ob_dim, "ac_dim", ac_dim)
    # algo = SAC()  # Run with defaults.
    # agent = SacAgent()
    adv_norm = True
    algo = A2C(learning_rate=lr,
               discount=0.99,  # 0.99
               gae_lambda=0.99,
               value_loss_coeff=0.5,  # 0.5
               entropy_loss_coeff=0.00,  # 0.01
               normalize_advantage=adv_norm,
               clip_grad_norm=1e6,
               )
    # algo.bootstrap_value = False
    model_kwargs = dict(ob_dim=ob_dim, ac_dim=ac_dim)
    agent = CategoricalPgAgent(
        ModelCls=ModelCls,
        model_kwargs=model_kwargs,
        initial_model_state_dict=None,
    )
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=log_steps,
        affinity=dict(cuda_idx=cuda_idx),
        seed=seed,
    )
    config = dict(env_id=env_id)
    name = "a2c_" + env_id
    log_dir = "example_2"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()

    # Close gym env, when using env.render()
    for env in sampler.collector.envs:
        env.close()

    traj_infos, eval_time = runner.evaluate_agent(n_steps)
    # runner.log_diagnostics(50e3, traj_infos, eval_time)
    rews = [x.Return for x in traj_infos]
    avg_rew = np.mean(rews)
    std_rew = np.std(rews)
    print("length of trajs: {}, avg_rew {}, std_rew {}".format(
        len(traj_infos), avg_rew, std_rew))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='CartPole-v0')
    # parser.add_argument('--env_id', help='environment ID', default='MountainCar-v0')
    # parser.add_argument('--env_id', help='environment ID', default='Hopper-v3')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
