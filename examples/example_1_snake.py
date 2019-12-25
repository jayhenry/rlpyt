
"""
Runs one instance of the Atari environment and optimizes using DQN algorithm.
Can use a GPU for the agent (applies to both sample and train). No parallelism
employed, so everything happens in one python process; can be easier to debug.

The kwarg snapshot_mode="last" to logger context will save the latest model at
every log point (see inside the logger for other options).

In viskit, whatever (nested) key-value pairs appear in config will become plottable
keys for showing several experiments.  If you need to add more after an experiment, 
use rlpyt.utils.logging.context.add_exp_param().

"""
import os.path as osp
from torch import nn
import torch
import numpy as np

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, to_onehot
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.logging import logger


from snake import SnakeEnv


def make(**kwargs):
    env = GymEnvWrapper(SnakeEnv(10, [3,6]))
    return env

class MyTrajInfo(TrajInfo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._traj = []

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self._traj.append((observation, action, reward, done))


class ModelCls(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ob_size = 100
        self.ac_size = 2
        self.model = MlpModel(self.ob_size*2, [128,128], self.ac_size)

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

        q = self.model(input.view(T * B, *input_shape))  # Fold if T dimension.

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


def build_and_train(game="pong", run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=make,
        TrajInfoCls=MyTrajInfo,
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=50,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=100,
    )
    # algo = SAC()  # Run with defaults.
    # agent = SacAgent()
    # runner = MinibatchRlEval(
    #     algo=algo,
    #     agent=agent,
    #     sampler=sampler,
    #     n_steps=1e6,
    #     log_interval_steps=1e4,
    #     affinity=dict(cuda_idx=cuda_idx),
    # )

    algo = DQN(batch_size=100, min_steps_learn=1e3)  # Run with defaults.
    # agent = AtariDqnAgent()
    agent = DqnAgent(ModelCls=ModelCls,
                    eps_eval=0,)
    runner = MinibatchRlEval(
       algo=algo,
       agent=agent,
       sampler=sampler,
       n_steps=50e3,
       log_interval_steps=1e3,
       affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        # runner.startup()
        # with logger.prefix(f"itr #0 "):
        #     eval_traj_infos, eval_time = runner.evaluate_agent(0)
        #     runner.log_diagnostics(0, eval_traj_infos, eval_time)
        #     print("Eval random policy by e-greedy")
        #     rews = [x.Return for x in eval_traj_infos]
        #     avg_rew = np.mean(rews)
        #     std_rew = np.std(rews)
        #     print("length of trajs:", len(eval_traj_infos), avg_rew, std_rew)
        #     # print(eval_traj_infos[0]._traj)
        runner.train()

    # # ckp_path = osp.abspath(osp.join(osp.dirname(__file__), '../data/local/20191224/example_1/run_1/params.pkl'))
    # ckp_path = osp.abspath(osp.join(osp.dirname(__file__), '../data/local/20191225/example_1/run_1/params.pkl'))
    # ckp = torch.load(ckp_path)
    # runner.agent.model.load_state_dict(ckp['agent_state_dict']['model'])
    # runner.agent.target_model.load_state_dict(ckp['agent_state_dict']['target'])

    # traj_infos, eval_time = runner.evaluate_agent(50e3)
    # runner.log_diagnostics(50e3, traj_infos, eval_time)
    # rews = [x.Return for x in traj_infos]
    # avg_rew = np.mean(rews)
    # std_rew = np.std(rews)
    # print("Eval trained Q-agent")
    # print("length of trajs:", len(traj_infos), avg_rew, std_rew)
    # # print(traj_infos)
    # # print(traj_infos[0]._traj)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='snake')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=1)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
