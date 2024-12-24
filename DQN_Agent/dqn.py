import torch.nn as nn
import torch

# Define the architecture for the neural network inside dqn
# use a CNN same as the Atari paper for deep q network
# In order to train, we need to have an image of size 84 * 84 (resized from 1000 * 390)
class DQN(nn.Module):
    def __init__(self, has_market_var: bool):
        super(DQN, self).__init__()
        # Current size (1 or 4, 84, 84)
        # Define first conv layer
        self.conv1 = nn.Conv2d(4, 16, kernel_size = (8, 8), stride = 4)
        self.act1 = nn.ReLU()
        # (16, 20, 20)
        # second conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size = (4, 4), stride = 2)
        self.act2 = nn.ReLU()
        # (32, 9, 9)
        # define flatten layer
        self.flat = nn.Flatten()
        # define fully-connected layer
        if has_market_var:
            self.fc = nn.Linear(32*9*9 + 2, 256)
        else:
            self.fc = nn.Linear(32*9*9, 256)
        self.act3 = nn.ReLU()
        # define output layer
        self.output = nn.Linear(256, 1000)

    def forward(self, x, m = None):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.flat(x)
        if m is not None:
            x = torch.cat([x, m], dim = 1)
        x = self.act3(self.fc(x))
        x = self.output(x)
        return x