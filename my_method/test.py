import argparse

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from utils import *
from trpo import one_step_trpo

from copy import deepcopy


a=torch.tensor([1.0,2.0],requires_grad=True)
b=(a*a).sum()
b.backward()

print(b)
print(a.grad.data)

a.grad.zero_()
print(a.grad.data)