import torch
import numpy as np
from trpo import one_step_trpo,conjugate_gradients
 
aa=torch.tensor(np.array([[1.0,1.0],[1.0,1.0]]),requires_grad=True)

bb=torch.special.expit(2.0*aa-2.0)*2
print(bb)