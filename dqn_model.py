import torch.nn as nn
import torch

class DQN(nn.Module):
    def __init__(self,action_size,observation_size):
        super().__init__()
        self.seq = nn.Sequential(
              nn.Linear(observation_size,64),
              nn.ReLU(),
              nn.Linear(64,64),
              nn.ReLU(),
              nn.Linear(64,128),
              nn.ReLU(),
              nn.Linear(128,action_size),
          )

    def forward(self,x):
      return self.seq(x)
        