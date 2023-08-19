import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class DQN(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

        #Parameters
        self.input_size = 17 * 17
        self.output_size = 6
        #Layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17*17, 121),
            nn.ReLU(),
            nn.Linear(121, 60),
            nn.ReLU(),
            nn.Linear(60, 6),
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output