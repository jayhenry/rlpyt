
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
from torch import nn
import torch

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


from snake import SnakeEnv


def make(**kwargs):
    env = GymEnvWrapper(SnakeEnv(10, [3,6]))
    return env

class ModelCls(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ob_size = 100
        self.ac_size = 2
        self.model = MlpModel(self.ob_size, [128,128], self.ac_size)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        # observation = observation.squeeze(-1)
        # print("ob:", observation)
        observation = to_onehot(observation, 100)
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
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(1e3),
        eval_max_trajectories=5,
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

    algo = DQN(batch_size=8, min_steps_learn=1e3)  # Run with defaults.
    # agent = AtariDqnAgent()
    agent = DqnAgent(ModelCls=ModelCls)
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
        runner.train()


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
