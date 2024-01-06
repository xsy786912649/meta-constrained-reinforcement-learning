import torch
import torch.autograd as autograd
import torch.nn as nn
import math

'''
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 128)
        self.affine2 = nn.Linear(128, 128)

        self.action_mean = nn.Linear(128, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = torch.zeros(1, num_outputs)

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std
'''

class Policy_new(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy_new, self).__init__()


        self.params=[torch.Tensor(128, num_inputs).uniform_(-1./math.sqrt(num_inputs), 1./math.sqrt(num_inputs)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(num_outputs, 128).uniform_(-0.1/math.sqrt(128), 0.1/math.sqrt(128)).requires_grad_(),
                    torch.Tensor(num_outputs).zero_().requires_grad_(),

                    ]
        
        self.action_log_std = torch.zeros(1, num_outputs)


    def forward(self, x):
        x=torch.nn.functional.linear(x, self.params[0], self.params[1])
        x = torch.tanh(x)
        x=torch.nn.functional.linear(x, self.params[2], self.params[3])
        x = torch.tanh(x)
        action_mean=torch.nn.functional.linear(x, self.params[4], self.params[5])

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std
    
    def model_forward(self, x, params):
        x=torch.nn.functional.linear(x, params[0], params[1])
        x = torch.tanh(x)
        x=torch.nn.functional.linear(x, params[2], params[3])
        x = torch.tanh(x)
        action_mean=torch.nn.functional.linear(x, params[4], params[5])

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std