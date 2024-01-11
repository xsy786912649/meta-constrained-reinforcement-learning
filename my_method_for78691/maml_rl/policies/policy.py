import torch
import torch.nn as nn

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

    def update_params(self, loss, params=None, step_size=0.5, first_order=True):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        #if params is None:
        #    params = OrderedDict(self.named_meta_parameters())

        params_meta = OrderedDict(self.named_meta_parameters())

        first_order = True
        for i in range(50):
            grads = torch.autograd.grad(loss, params.values(),  retain_graph=True,
                                        create_graph=not first_order)
            print(loss)
            
            updated_params = OrderedDict()
            for (name, param), (name_meta, param_meta), grad in zip(params.items(), params_meta.items(), grads):
                updated_params[name] = param_meta - 0.3 * step_size * grad 
                param = param - 0.3 * step_size * grad 

        return updated_params
