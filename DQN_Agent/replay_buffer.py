from collections import deque
import numpy as np
import torch
from .utils import *

# Define the replay buffer to store all randomized + explored state + action + rewards => use in batch to train the model
class ReplayBuffer:
    def __init__(self, num_shares, num_time_units, has_market_var: bool = False, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.num_shares = num_shares 
        self.num_time_units = num_time_units
        self.has_market_var = has_market_var

    def add(self, experience: tuple):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, market_vars, actions, rewards, next_states, next_market_vars, dones = zip(*[self.buffer[idx] for idx in indices])
        # Convert states to matrix
        i_states = numeric_state_to_image_state(states, self.num_shares, self.num_time_units)
        i_states = resize_image_state(i_states)
        # Convert states to image
        i_next_states = numeric_state_to_image_state(next_states, self.num_shares, self.num_time_units)
        i_next_states = resize_image_state(i_next_states)
        # Process in the case if we have or not have market variable
        if self.has_market_var:
            market_vars = list(map(lambda x: list(x.values()), market_vars))
            next_market_vars = list(map(lambda x: list(x.values()), next_market_vars))
            return (
                i_states,
                torch.tensor(market_vars, dtype=torch.int64),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                i_next_states,
                torch.tensor(next_market_vars, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32),
            )
        else:
            return (
                i_states,
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                i_next_states,
                torch.tensor(dones, dtype=torch.float32),
            )

    def size(self):
        return len(self.buffer)