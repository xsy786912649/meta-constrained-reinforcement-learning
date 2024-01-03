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
from trpo import one_step_trpo,conjugate_gradients

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
parser.add_argument('--meta-lambda', type=float, default=2.0, metavar='G', 
                    help='meta meta-lambda (default: 0.5)')  # 0.5
parser.add_argument('--max-kl', type=float, default=3e-2, metavar='G',
                    help='max kl value (default: 3e-2)')
parser.add_argument('--damping', type=float, default=0e-1, metavar='G',
                    help='damping (default: 0e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='batch-size (default: 20)') #20  
parser.add_argument('--task-batch-size', type=int, default=5, metavar='N',
                    help='task-batch-size (default: 5)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--max-length', type=int, default=200, metavar='N',
                    help='max length of a path (default: 200)')
parser.add_argument('--lower-opt', type=str, default="Adam", metavar='N',
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
running_reward = ZFilter((1,), demean=False, clip=10)

def setting_reward():
    return np.random.uniform(0.0,2.0)

def select_action(state,policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

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

def task_specific_adaptation(task_specific_policy,meta_policy_net_copy,batch,q_values,index=1): 
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))

    action_means, action_log_stds, action_stds = meta_policy_net_copy(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss():
        action_means1, action_log_stds1, action_stds1 = task_specific_policy(Variable(states))
        log_prob = normal_log_density(Variable(actions), action_means1, action_log_stds1, action_stds1)
        aaaa=torch.exp(log_prob - Variable(fixed_log_prob))
        action_loss = -Variable(q_values) *  torch.special.expit(2.0*aaaa-2.0)*2 
        #action_loss = -Variable(q_values) * aaaa
        return action_loss.mean()     

    def get_kl():
        mean1, log_std1, std1 = task_specific_policy(Variable(states))
        mean_previous, log_std_previous, std_previous = meta_policy_net_copy(Variable(states))

        mean0 = mean_previous.clone().detach().data.double()
        log_std0 = log_std_previous.clone().detach().data.double()
        std0 = std_previous.clone().detach().data.double()

        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    def get_kl2():
        mean1, log_std1, std1 = task_specific_policy(Variable(states))
        mean_previous, log_std_previous, std_previous = meta_policy_net_copy(Variable(states))

        mean0 = mean_previous.clone().detach().data.double()
        log_std0 = log_std_previous.clone().detach().data.double()
        std0 = std_previous.clone().detach().data.double()

        kl = log_std0 - log_std1 + (std1.pow(2) + (mean1 - mean0).pow(2)) / (2.0 * std0.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    def get_kl3():
        policy_dictance=torch.tensor(0.0,requires_grad=True)
        for i,param in enumerate(task_specific_policy.parameters()):
            policy_dictance += (param-list(meta_policy_net_copy.parameter())[i].clone().detach().data).pow(2).sum() 
        return policy_dictance
    if index==1:
        one_step_trpo(task_specific_policy, get_loss, get_kl,args.meta_lambda,args.lower_opt) 
    elif index==2:
        one_step_trpo(task_specific_policy, get_loss, get_kl2,args.meta_lambda,args.lower_opt) 
    elif index==3:
        one_step_trpo(task_specific_policy, get_loss, get_kl3,args.meta_lambda3,args.lower_opt) 

    return task_specific_policy

def kl_divergence(meta_policy_net1,task_specific_policy1,batch,index=1):
    if index==1:
        states = torch.Tensor(np.array(batch.state))
        mean1, log_std1, std1 = task_specific_policy1(Variable(states))
        mean0, log_std0, std0 = meta_policy_net1(Variable(states))
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True).mean()
    elif index==2:
        states = torch.Tensor(np.array(batch.state))
        mean1, log_std1, std1 = task_specific_policy1(Variable(states))
        mean0, log_std0, std0 = meta_policy_net1(Variable(states))
        kl = log_std0 - log_std1 + (std1.pow(2) + (mean1 - mean0).pow(2)) / (2.0 * std0.pow(2)) - 0.5
        return kl.sum(1, keepdim=True).mean()
    elif index==3:
        policy_dictance=torch.tensor(0.0,requires_grad=True)
        for param,param1 in zip(task_specific_policy1.parameters(),meta_policy_net1.parameters()):
            policy_dictance += (param-param1).pow(2).sum() 
        return policy_dictance

def policy_gradient_obain(task_specific_policy,after_batch,after_q_values):
    actions = torch.Tensor(np.array(np.concatenate(after_batch.action, 0)))
    states = torch.Tensor(np.array(after_batch.state))
    fixed_action_means, fixed_action_log_stds, fixed_action_stds = task_specific_policy(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), fixed_action_means, fixed_action_log_stds, fixed_action_stds).data.clone()
    afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds = task_specific_policy(Variable(states))
    log_prob = normal_log_density(Variable(actions), afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds)
    AAAAA=torch.exp(log_prob - Variable(fixed_log_prob))
    bbbbb=torch.min(Variable(after_q_values)*AAAAA,Variable(after_q_values)*AAAAA*torch.clamp(AAAAA,0.8,1.2))
    J_loss = (-bbbbb).mean()
    for param in task_specific_policy.parameters():
        param.grad.zero_()
    J_loss.backward(retain_graph=False)
    policy_grad = [param2.grad.data.clone() for param2 in task_specific_policy.parameters()]

    return J_loss, policy_grad


def loss_obain_new(task_specific_policy,meta_policy_net_copy,after_batch,after_q_values):
    actions = torch.Tensor(np.array(np.concatenate(after_batch.action, 0)))
    states = torch.Tensor(np.array(after_batch.state))
    fixed_action_means, fixed_action_log_stds, fixed_action_stds = meta_policy_net_copy(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), fixed_action_means, fixed_action_log_stds, fixed_action_stds).detach().data.clone()
    afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds = task_specific_policy(Variable(states))
    log_prob = normal_log_density(Variable(actions), afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds)
    aaaaa=torch.exp(log_prob - Variable(fixed_log_prob))
    J_loss = (-Variable(after_q_values) * torch.special.expit(2.0*aaaaa-2.0)*2 ).mean()
    #J_loss = (-Variable(after_q_values) * torch.exp(log_prob - Variable(fixed_log_prob))).mean()
    
    return J_loss


if __name__ == "__main__":

    if not os.path.exists("./meta_policy_net.pkl"):
        meta_policy_net = Policy(num_inputs, num_actions)
    else:
        meta_policy_net = torch.load("meta_policy_net.pkl")

    "--------------------------------------------------for initialization of running_state------------------------------------------"
    for i in range(args.batch_size):
        state = env.reset()[0]
        state = running_state(state)
        for t in range(args.max_length):
            action = select_action(state,meta_policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            next_state = running_state(next_state)

    optimizer = torch.optim.Adam(meta_policy_net.parameters(), lr=0.003)

    for i_episode in range(500):
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

            task_specific_policy=Policy(num_inputs, num_actions)
            meta_policy_net_copy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            for i,param in enumerate(meta_policy_net_copy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation(task_specific_policy,meta_policy_net_copy,batch,q_values1,index=1)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size)
            print('(after adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, after_accumulated_raward_batch)) 

            q_values_after = compute_adavatage(after_batch,after_batch_extra,args.batch_size)
            q_values_after = (q_values_after - q_values_after.mean()) 

            kl_phi_theta=kl_divergence(meta_policy_net,task_specific_policy,batch,index=1)

            _, policy_gradient_main_term= policy_gradient_obain(task_specific_policy,after_batch,q_values_after)

            loss_for_1term=loss_obain_new(task_specific_policy,meta_policy_net,batch,q_values1)
            
            #(\nabla_\phi^2 kl_phi_theta+loss_for_1term) x= policy_gradient_2term
            def d_theta_2_kl_phi_theta_loss_for_1term(v):
                grads = torch.autograd.grad(kl_phi_theta+loss_for_1term/args.meta_lambda, task_specific_policy.parameters(), create_graph=True,retain_graph=True)
                flat_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
                kl_v = (flat_grad_kl * Variable(v)).sum()
                grads_new = torch.autograd.grad(kl_v, task_specific_policy.parameters(), create_graph=True,retain_graph=True)
                flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads_new]).data.clone()
                return flat_grad_grad_kl
            policy_gradient_main_term_flat=torch.cat([grad.contiguous().view(-1) for grad in policy_gradient_main_term]).data
            x = conjugate_gradients(d_theta_2_kl_phi_theta_loss_for_1term, policy_gradient_main_term_flat, 10)

            kl_phi_theta_1=kl_divergence(meta_policy_net,task_specific_policy,batch_2,index=1)
            grads_1 = torch.autograd.grad(kl_phi_theta_1, task_specific_policy.parameters(), create_graph=True,retain_graph=True)
            flat_grad_kl_1 = torch.cat([grad.contiguous().view(-1) for grad in grads_1])
            kl_v_1 = (flat_grad_kl_1 * x.data).sum() 
            grads_new_1 = torch.autograd.grad(kl_v_1, meta_policy_net.parameters(), create_graph=True,retain_graph=True)
            if grads_update==None:
                grads_update=[grad.clone().data*1.0/args.task_batch_size for grad in grads_new_1]
            else:
                grads_update=[grads_update[i]+ grad.clone().data*1.0/args.task_batch_size for i,grad in enumerate(grads_new_1)]

        optimizer.zero_grad() 
        for i,param in enumerate(meta_policy_net.parameters()): 
            param.grad= -grads_update[i]
        optimizer.step()
        optimizer.zero_grad()

        meta_value_net_copy = Value(num_inputs)
        for i,param in enumerate(meta_value_net_copy.parameters()):
            param.data.copy_(list(meta_value_net.parameters())[i].clone().detach().data)
        meta_value_net=update_meta_valuenet(meta_value_net,meta_value_net_copy,data_pool_for_meta_value_net,args.batch_size)

        torch.save(meta_policy_net, "meta_policy_net.pkl")
        torch.save(meta_value_net, "meta_value_net.pkl")

        for task_number_test in range(1):
            target_v=1.2
            batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
            print("test_task: ", " target_v: ", target_v)
            print('(before adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, accumulated_raward_batch))
    
            task_specific_value_net = Value(num_inputs)
            meta_value_net_copy = Value(num_inputs)
            for i,param in enumerate(task_specific_value_net.parameters()):
                param.data.copy_(list(meta_value_net.parameters())[i].clone().detach().data)
            for i,param in enumerate(meta_value_net_copy.parameters()):
                param.data.copy_(list(meta_value_net.parameters())[i].clone().detach().data)
            task_specific_value_net = update_task_specific_valuenet(task_specific_value_net,meta_value_net_copy,batch,batch_extra,args.batch_size)
            
            batch_2,batch_extra_2,_=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
            q_values = compute_adavatage(batch_2,batch_extra_2,args.batch_size)
            q_values = (q_values - q_values.mean())  

            task_specific_policy=Policy(num_inputs, num_actions)
            meta_policy_net_copy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            for i,param in enumerate(meta_policy_net_copy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation(task_specific_policy,meta_policy_net_copy,batch_2,q_values,index=1)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size)
            print('(after adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, after_accumulated_raward_batch)) 
            print("-----------------------------------------")

