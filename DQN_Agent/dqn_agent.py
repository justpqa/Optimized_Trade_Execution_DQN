import random
from typing import Any
import torch
from torch import nn
import torch.optim as optim
from .dqn import DQN
from .replay_buffer import ReplayBuffer
from .utils import *

# Define the Deep Q Network agent with a network and target network
class DQNAgent:
    def __init__(self, device, num_time_units: int, num_shares: int, has_market_var: bool = False, gamma=0.99, epsilon=0.6, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001):
        self.device = device
        self.num_time_units = num_time_units
        self.num_shares = num_shares
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.dqn = DQN(has_market_var).to(device)
        self.target_dqn = DQN(has_market_var).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(num_shares, num_time_units, has_market_var)

        self.has_market_var = has_market_var
        
    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def select_action(self, state: tuple, market_var: dict[str, Any]):
        curr_num_shares = state[1] % (self.num_shares + 1)
        if state[0] == self.num_time_units - 1:
            return curr_num_shares
        elif curr_num_shares == 0:
            return 0
        elif random.random() < self.epsilon:
            return random.randint(0, curr_num_shares)
        else:
            with torch.no_grad():
                i_state = numeric_state_to_image_state([state], self.num_shares, self.num_time_units)
                i_state = torch.tensor(i_state, dtype=torch.float32)
                i_state = resize_image_state(i_state).to(self.device)
                if self.has_market_var:
                    market_var = list(market_var.values())
                    market_var = torch.tensor(market_var, dtype=torch.float32)
                    market_var = market_var.unsqueeze(0)
                    market_var = market_var.to(self.device)
                    q_values = self.dqn(i_state, market_var)
                    q_values = q_values[0, :curr_num_shares + 1]
                    return torch.argmax(q_values).item()
                else:
                    q_values = self.dqn(i_state)
                    q_values = q_values[0, :curr_num_shares + 1]
                    return torch.argmax(q_values).item()
    
    def select_best_action(self, state: tuple, market_var: dict[str, Any]):
        curr_num_shares = state[1] % (self.num_shares + 1)
        if state[0] == self.num_time_units - 1:
            return curr_num_shares
        elif curr_num_shares == 0:
            return 0
        else:
            with torch.no_grad():
                i_state = numeric_state_to_image_state([state], self.num_shares, self.num_time_units)
                i_state = torch.tensor(i_state, dtype=torch.float32)
                i_state = resize_image_state(i_state).to(self.device)
                if self.has_market_var:
                    market_var = list(market_var.values())
                    market_var = torch.tensor(market_var, dtype=torch.float32)
                    market_var = market_var.unsqueeze(0)
                    market_var = market_var.to(self.device)
                    q_values = self.dqn(i_state, market_var)
                    q_values = q_values[0, :curr_num_shares + 1]
                    return torch.argmax(q_values).item()
                else:
                    q_values = self.dqn(i_state)
                    q_values = q_values[0, :curr_num_shares + 1]
                    return torch.argmax(q_values).item()

    def train(self, batch_size: int = 8):
        if self.replay_buffer.size() < batch_size:
            return

        if self.has_market_var:
            i_states, market_vars, actions, rewards, i_next_states, next_market_vars, dones = self.replay_buffer.sample(batch_size)
            i_states = i_states.to(self.device)
            market_vars = market_vars.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            i_next_states = i_next_states.to(self.device)
            next_market_vars = next_market_vars.to(self.device)
            dones = dones.to(self.device)
        else:
            i_states, actions, rewards, i_next_states, dones = self.replay_buffer.sample(batch_size)
            i_states = i_states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            i_next_states = i_next_states.to(self.device)
            dones = dones.to(self.device)

        # Calculate target Q-values
        with torch.no_grad():
            if self.has_market_var:
                next_q_values = self.target_dqn(i_next_states, next_market_vars)
            else:
                next_q_values = self.target_dqn(i_next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Calculate current Q-values
        if self.has_market_var:
            q_values = self.dqn(i_states, market_vars)
        else:
            q_values = self.dqn(i_states)
        action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the loss
        loss = nn.MSELoss()(action_q_values, target_q_values)

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_agent_model(self, dir_path: str):
        # Save the model
        torch.save(self.dqn.state_dict(), f"{dir_path}/dqn.pth")
        torch.save(self.target_dqn.state_dict(), f"{dir_path}/target_dqn.pth")
    
    def load_agent_model(self, dqn_path: str, target_dqn_path: str):
        self.dqn.load_state_dict(torch.load(dqn_path)) 
        self.target_dqn.load_state_dict(torch.load(target_dqn_path))