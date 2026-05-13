import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class loss_fn(nn.Module):

    def __init__(self,configs):
        super(loss_fn, self).__init__()
        self.label_idx = configs['label_idx']
        self.constraints = self._get_constraints()

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def forward(self, model, input, label, epoch):
        
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 
        output = model.model(input, label, self.constraints, teacher_forcing_ratio)

        return output['loss'], output
    