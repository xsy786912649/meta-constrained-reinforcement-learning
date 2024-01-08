import argparse
import os
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from utils import *

import pickle

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
parser.add_argument('--meta-reg', type=float, default=0.001, metavar='G',
                    help='meta regularization regression (default: 1.0)') 
parser.add_argument('--meta-lambda', type=float, default=10.0, metavar='G', 
                    help='meta meta-lambda (default: 0.5)')  # 0.5
parser.add_argument('--max-kl', type=float, default=3e-2, metavar='G',
                    help='max kl value (default: 3e-2)')
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
parser.add_argument('--lower-opt', type=str, default="PG", metavar='N',
                    help='lower-opt (default: Adam)')
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

reference_policy_net=Policy_new(num_inputs, num_actions)

def setting_reward():
    return np.random.uniform(0.0,2.0)

def select_action(state,policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def select_action_test(state,policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    return action_mean


def sample_data_for_task_specific(target_v,policy_net,batch_size):
    memory = Memory()
    memory_extra=Memory()

    accumulated_raward_batch = 0
    num_episodes = 0
    for i in range(batch_size):
        state = env.reset()[0]
        state = running_state(state)

        reward_sum = 0
        for t in range(args.max_length):
            action = select_action(state,policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            reward=-abs(info['x_velocity']-target_v)#-0.5 * 1e-1 * np.sum(np.square(action))
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
            reward=-abs(info['x_velocity']-target_v)#-0.5 * 1e-1 * np.sum(np.square(action))
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


def compute_adavatage(batch,batch_extra,batch_size):
    rewards = torch.Tensor(np.array(batch.reward))
    path_numbers = torch.Tensor(np.array(batch.path_number))
    actions = torch.Tensor(np.array(np.concatenate(batch.action, 0)))
    states = torch.Tensor(np.array(batch.state))

    rewards_extra = torch.Tensor(np.array(batch_extra.reward))
    path_numbers_extra = torch.Tensor(np.array(batch_extra.path_number))
    actions_extra = torch.Tensor(np.array(np.concatenate(batch_extra.action, 0)))
    states_extra = torch.Tensor(np.array(batch_extra.state))

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
    return targets

def task_specific_adaptation(meta_policy_net,meta_policy_net_copy,batch,q_values,index):

    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))

    action_means, action_log_stds, action_stds = meta_policy_net_copy(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    action_means1, action_log_stds1, action_stds1 = meta_policy_net(Variable(states))
    log_prob = normal_log_density(Variable(actions), action_means1, action_log_stds1, action_stds1)
    aaaa=torch.exp(log_prob - Variable(fixed_log_prob))
    action_loss = -Variable(q_values) *  torch.special.expit(2.0*aaaa-2.0)*2 

    grads=torch.autograd.grad(action_loss.mean() , meta_policy_net.params, create_graph=True,retain_graph=True)
    [grads[i].retain_grad() for i in range(len(grads))]
    task_specific_policy_parameter = [list(meta_policy_net.params)[i] - 1.0/args.meta_lambda * grads[i] for i in range(len(list(meta_policy_net.params)))]
    [task_specific_policy_parameter[i].retain_grad() for i in range(len(task_specific_policy_parameter))]

    return task_specific_policy_parameter

def policy_gradient_loss_obain(task_specific_policy_parameter,after_batch,after_q_values):
    
    actions = torch.Tensor(np.array(np.concatenate(after_batch.action, 0)))
    states = torch.Tensor(np.array(after_batch.state))
    fixed_action_means, fixed_action_log_stds, fixed_action_stds = reference_policy_net.model_forward(Variable(states), task_specific_policy_parameter)
    fixed_log_prob = normal_log_density(Variable(actions), fixed_action_means, fixed_action_log_stds, fixed_action_stds).detach().clone().data
    afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds = reference_policy_net.model_forward(Variable(states), task_specific_policy_parameter)
    log_prob = normal_log_density(Variable(actions), afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds)
    AAAAA=torch.exp(log_prob - Variable(fixed_log_prob))
    #bbbbb=torch.min(Variable(after_q_values)*AAAAA,Variable(after_q_values)*AAAAA*torch.clamp(AAAAA,0.8,1.2))
    bbbbb=Variable(after_q_values)*torch.special.expit(2.0*AAAAA-2.0)*2 
    J_loss = (-bbbbb).mean()   

    return J_loss


index = 1
model_lower="PG"

if __name__ == "__main__":
    if not os.path.exists("meta_policy_net_"+model_lower+".pkl"):
        meta_policy_net = Policy_new(num_inputs, num_actions)
    else:
        meta_policy_net = torch.load("meta_policy_net_"+model_lower+".pkl")

    "--------------------------------------------------for initialization of running_state------------------------------------------"
    for i in range(args.batch_size*5):
        state = env.reset()[0]
        state = running_state(state)
        for t in range(args.max_length):
            action = select_action(state,meta_policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            next_state = running_state(next_state)

    optimizer = torch.optim.Adam(meta_policy_net.params, lr=0.003)

    for i_episode in range(1000):
        print("i_episode: ",i_episode)
        grads_update=None
        for task_number in range(args.task_batch_size):
            target_v=setting_reward()
            batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
            print("task_number: ",task_number, " target_v: ", target_v)
            print('(before adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, accumulated_raward_batch))
            
            q_values = compute_adavatage(batch,batch_extra,args.batch_size)
            q_values2 = q_values
            q_values1 = (q_values - q_values.mean()) 

            meta_policy_net_copy=Policy_new(num_inputs, num_actions)
            for i,param in enumerate(meta_policy_net_copy.params):
                param.data.copy_(list(meta_policy_net.params)[i].clone().detach().data)
            task_specific_policy_parameter=task_specific_adaptation(meta_policy_net,meta_policy_net_copy,batch,q_values1,index)

            task_specific_policy=Policy_new(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.params):
                param.data.copy_(task_specific_policy_parameter[i].clone().detach().data)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size*5) 
            print('(after adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, after_accumulated_raward_batch)) 

            q_values_after = compute_adavatage(after_batch,after_batch_extra,args.batch_size*5) 
            q_values_after = (q_values_after - q_values_after.mean()) 

            J_loss= policy_gradient_loss_obain(task_specific_policy_parameter,after_batch,q_values_after)

            grads_new_1=torch.autograd.grad(J_loss, meta_policy_net.params)
            
            if grads_update==None:
                grads_update=[grad.clone().data*1.0/args.task_batch_size for grad in grads_new_1]
            else:
                grads_update=[grads_update[i]+ grad.clone().data*1.0/args.task_batch_size for i,grad in enumerate(grads_new_1)]

        optimizer.zero_grad() 
        for i,param in enumerate(meta_policy_net.params): 
            param.grad= -grads_update[i]
        optimizer.step()
        optimizer.zero_grad()

        torch.save(meta_policy_net, "meta_policy_net_"+model_lower+".pkl")
        #torch.save(meta_policy_net, "./check_point/meta_policy_net_"+model_lower+"_"+str(i_episode)+".pkl")

        target_v_list000=[0.3,1.0,1.7]
        result_before=np.zeros(3)
        result_after=np.zeros(3)
        for task_number_test in range(3):
            target_v=target_v_list000[task_number_test]
            batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
            result_before[task_number_test]=accumulated_raward_batch
    
            q_values = compute_adavatage(batch,batch_extra,args.batch_size)
            q_values = (q_values - q_values.mean())  

            meta_policy_net_copy=Policy_new(num_inputs, num_actions)
            for i,param in enumerate(meta_policy_net_copy.params):
                param.data.copy_(list(meta_policy_net.params)[i].clone().detach().data)
            
            task_specific_policy_parameter=task_specific_adaptation(meta_policy_net,meta_policy_net_copy,batch,q_values,index)

            task_specific_policy=Policy_new(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.params):
                param.data.copy_(task_specific_policy_parameter[i].clone().detach().data)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size)
            result_after[task_number_test]=after_accumulated_raward_batch

        print("result_before: ",result_before.mean())
        print("result_after: ",result_after.mean())
        
        output_hal = open("running_state_"+model_lower+".pkl", 'wb')
        str1 = pickle.dumps(running_state)
        output_hal.write(str1)
        output_hal.close()
        
        print(torch.exp(meta_policy_net.action_log_std)) 

