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
parser.add_argument('--meta-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=0e-1, metavar='G',
                    help='damping (default: 0e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch-size (default: 20)')
parser.add_argument('--task-batch-size', type=int, default=5, metavar='N',
                    help='task-batch-size (default: 5)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--max-length', type=int, default=200, metavar='N',
                    help='max length of a path (default: 200)')
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

def setting_reward():
    return np.random.uniform(0.0,2.0)

def select_action(state,policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def sample_data_for_task_specific(target_v,policy_net):
    memory = Memory()
    memory_extra=Memory()

    accumulated_raward_batch = 0
    num_episodes = 0
    for i in range(args.batch_size):
        state = env.reset()[0]
        state = running_state(state)

        reward_sum = 0
        for t in range(args.max_length):
            action = select_action(state,policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            reward=-abs(info['x_velocity']-target_v)
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
            action = select_action(state,policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info= env.step(action)
            reward=-abs(info['x_velocity']-target_v)
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

    return batch,batch_extra,accumulated_raward_batch

def update_task_specific_valuenet(value_net,meta_value_net,batch,batch_extra,batch_size):

    rewards = torch.Tensor(batch.reward)
    path_numbers = torch.Tensor(batch.path_number)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)

    rewards_extra = torch.Tensor(batch_extra.reward)
    path_numbers_extra = torch.Tensor(batch_extra.path_number)
    actions_extra = torch.Tensor(np.concatenate(batch_extra.action, 0))
    states_extra = torch.Tensor(batch_extra.state)

    returns = torch.Tensor(actions.size(0),1)
    prev_return=torch.zeros(batch_size,1)

    k=batch_size-1
    for i in reversed(range(rewards_extra.size(0))):
        if not int(path_numbers_extra[i].item())==k:
            k=k-1
            assert k==path_numbers_extra[i].item()
        prev_return[k,0]=rewards[i]+ args.gamma * prev_return[k,0] 
    
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return[int(path_numbers[i].item()),0]
        prev_return[int(path_numbers[i].item()),0] = returns[i, 0]
    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))
        value_loss = (values_ - targets).pow(2).mean()

        for i,param in enumerate(value_net.parameters()):
            value_loss += (param-list(meta_value_net.parameter())[i].data).pow(2).sum() * args.meta_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(meta_value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    return value_net


if __name__ == "__main__":

    meta_policy_net = Policy(num_inputs, num_actions)
    meta_value_net = Value(num_inputs)

    for i_episode in range(200):

        for task_number in range(args.task_batch_size):
            target_v=setting_reward()
            batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net)

            print('Episode {}\tAverage reward {:.2f}'.format(i_episode, accumulated_raward_batch))
    
            task_specific_value_net = Value(num_inputs)
            for i,param in enumerate(task_specific_value_net.parameters()):
                param.data.copy_(list(meta_value_net.parameters())[i].data)
            task_specific_value_net = update_task_specific_valuenet(task_specific_value_net,meta_value_net,batch,batch_extra,args.batch_size)

            compute_adavatage()

            task_specific_adaptation()

            task_meta_gradient_computation()

        update_meta_policy()

        update_meta_valuenet()

