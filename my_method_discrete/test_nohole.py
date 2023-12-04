from train_nohole import *

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from extra_function_nohole import *
import random
from meta_policy import *


if dis_i==2:
    lambda1 =0.25
elif dis_i==1:
    lambda1 =1.0

total_rewards_alltask_0=[]
total_rewards_alltask_1=[]
total_rewards_alltask_2=[]
total_rewards_alltask_3=[]
total_rewards_alltask_4=[]

for num_tasks in range(task_number):

    reward_map = np.load('maps_nohole/map'+str(num_tasks)+'.npy')
    env = gym.make("FrozenLake-v1",desc= ['SFFF', 'FFFF', 'FFFF', 'FFFG'], is_slippery=False)

    task_specific_theta = training_meta_theta_no_hole
    total_rewards=[]
    print("--------------")
    for i in range(5):
        total_reward,observations,qtable_meta_policy = sample_trajectorie(reward_map,env, gamma, task_specific_theta)
        print(i," times update: ","total_reward: ",total_reward)
        task_specific_policy=one_step_adaptation(task_specific_theta,qtable_meta_policy,lambda1,dis_i) 
        task_specific_theta=torch.log(task_specific_policy) 
        total_rewards.append(total_reward)
    total_rewards_alltask_0.append(total_rewards[0])
    total_rewards_alltask_1.append(total_rewards[1])
    total_rewards_alltask_2.append(total_rewards[2])
    total_rewards_alltask_3.append(total_rewards[3])
    total_rewards_alltask_4.append(total_rewards[4])

print(np.average(np.array(total_rewards_alltask_0)))
print(np.average(np.array(total_rewards_alltask_1)))
print(np.average(np.array(total_rewards_alltask_2)))
print(np.average(np.array(total_rewards_alltask_3)))
print(np.average(np.array(total_rewards_alltask_4)))