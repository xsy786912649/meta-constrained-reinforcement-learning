import torch
import torch.nn as nn

from maml_rl.utils.torch_utils import weighted_mean, to_numpy, detach_distribution
from torch.distributions.kl import kl_divergence

from collections import OrderedDict

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # For compatibility with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_params(self, reinforce_loss, episodes, policy, params=None, step_size=0.5, first_order=True):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        #if params is None:
        #    params = OrderedDict(self.named_meta_parameters())

        params_meta = OrderedDict(self.named_meta_parameters())

        pi1 = policy(episodes.observations, params=params)
        old_pi1 = detach_distribution(pi1)

        first_order = True
        for i in range(50):
            inner_loss = reinforce_loss(policy,episodes,params=params)
            
            kls = weighted_mean(kl_divergence(policy(episodes.observations, params=params), old_pi1),lengths=episodes.lengths)
            
            grads = torch.autograd.grad(inner_loss+kls.mean()*0.5, params.values())
            
            updated_params = OrderedDict()
            for (name, param), (name_meta, param_meta), grad in zip(params.items(), params_meta.items(), grads):
                updated_params[name] = param_meta - 0.3 * step_size * grad 
                param.data = param.data - 0.3 * step_size * grad 

        return updated_params
