import torch
import numpy as np
from trpo import one_step_trpo,conjugate_gradients
 
aa=torch.tensor(np.array([[1.0,0.3],[0.0,2.0]]))
bb=torch.tensor(np.array([1.0,2.0]))

def assfsfe(v):
    return aa@v
b = conjugate_gradients(assfsfe, bb, 10)
print(b )

