import torch
import numpy as np


class UserEncode(torch.nn.Module):
    def __init__(self, input_dim):
        super(UserEncode, self).__init__()
        # Define a single linear layer that maps the input dimension to 64 dimensions
        self.fc1 = torch.nn.Linear(input_dim, 64)
    
    def forward(self, x):
        # Pass the input through the linear layer
        
        # Check if x is a numpy array and convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # Ensure the input tensor has the same dtype and device as the model parameters
        x = x.to(self.fc1.weight.device).to(self.fc1.weight.dtype)
        
        x = self.fc1(x)
        return x

class ItemEncode(torch.nn.Module):
    def __init__(self, input_dim):
        super(ItemEncode, self).__init__()
        # Define a single linear layer that maps the input dimension to 64 dimensions
        self.fc1 = torch.nn.Linear(input_dim, 64)
    
    def forward(self, x):
        # Pass the input through the linear layer

        # Check if x is a numpy array and convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # Ensure the input tensor has the same dtype and device as the model parameters
        x = x.to(self.fc1.weight.device).to(self.fc1.weight.dtype)
        
        x = self.fc1(x)
        return x

