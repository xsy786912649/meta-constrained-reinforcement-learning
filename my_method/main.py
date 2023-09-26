import argparse

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from utils import *

from copy import deepcopy

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="HalfCheetah-v4", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=0e-1, metavar='G',
                    help='damping (default: 0e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=25, metavar='N',
                    help='batch-size (default: 25)')
parser.add_argument('--task-batch-size', type=int, default=5, metavar='N',
                    help='task-batch-size (default: 5)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--max-length', type=int, default=10000, metavar='N',
                    help='max length of a path (default: 10000)')
args = parser.parse_args()

torch.manual_seed(args.seed)
#if args.env_name=="HalfCheetah-v4":
#    env = gym.make(args.env_name,exclude_current_positions_from_observation=False)
#else:
#    env = gym.make(args.env_name)
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

def select_action(state,policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def sample_data_for_task_specific(reward_setting):
    memory = Memory()
    memory_extra=Memory()

    accumulated_raward_batch = 0
    num_episodes = 0
    for i in range(args.batch_size):
        state = env.reset()[0]
        state = running_state(state)

        reward_sum = 0
        for t in range(args.max_length):
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            reward_sum += reward
            next_state = running_state(next_state)
            path_number = i

            memory.push(state, np.array([action]), path_number, next_state, reward)
            if args.render:
                env.render()
            state = next_state
            if done or truncated:
                break
    
        env._elapsed_steps=0
        for t in range(args.max_length):
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info= env.step(action)
            next_state = running_state(next_state)
            path_number = i

            memory_extra.push(state, np.array([action]), path_number, next_state, reward)
            if args.render:
                env.render()
            state = next_state
            if done or truncated:
                break

        num_episodes += 1
        accumulated_raward_batch += reward_sum

    accumulated_raward_batch /= num_episodes
    batch = memory.sample()
    batch_extra = memory_extra.sample()

    return batch,batch_extra,accumulated_raward_batch,accumulated_raward_batch



if __name__ == "__main__":
    
    meta_policy_net = Policy(num_inputs, num_actions)
    meta_value_net = Value(num_inputs)

    for i_episode in range(200):

        for task_number in range(args.task_batch_size):
            reward_setting=setting_reward()
            batch,batch_extra,accumulated_raward_batch,accumulated_raward_batch=sample_data_for_task_specific(reward_setting)
            print('Episode {}\tAverage reward {:.2f}'.format(i_episode, accumulated_raward_batch))
    
        update_task_specific_valuenet_and_adavatage()

        task_specific_adaptation()

        update_meta_valuenet()
