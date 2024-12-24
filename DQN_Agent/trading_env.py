from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Discrete, Tuple
from .orderbook_env import OrderBookEnv

# Create custom environment for training Deep Q Network
class TradingEnv(gym.Env):
    def __init__(self, num_shares: int, num_time_units: int, num_date: int, data: pd.DataFrame, has_market_var: bool = False):
        # Define number of shares we have at the beginning and number of time units => range of states
        self.ns = num_shares
        self.num_time_units = num_time_units
        
        # Define the current date that we are using for the current training
        self.current_date = 0 # default is the first date in the training data
        self.num_date = num_date # The number of date that we can get from our training data
        
        # Define the current order book environment
        self.has_market_var = has_market_var
        self.orderbook = OrderBookEnv(data, has_market_var)
        
        # Define the current state that we are at
        # We will use the state to be the last 4 # of shares with their associated timestamps
        # (if num shares small, can make this into a 4-digit number of base num_shares + 1)
        self._last_4_num_shares = self.ns
        self._current_time = 0
        
        # Define the action space: number of stock for market order
        self.action_space = Discrete(self.ns + 1)
        
        # Define the observation space: 
        # current time out of the 390 time units of a trading day + number of shares left
        max_state = self.ns * ((self.ns + 1) ** 3 + (self.ns + 1) ** 2 + (self.ns + 1) + 1)
        self.observation_space = Tuple((Discrete(self.num_time_units), Discrete(max_state)))
    
    def set_current_date(self, date_inx: int):
        self.current_date = date_inx
        return

    def get_market_var(self):
        if self.has_market_var:
            return self.orderbook.compute_market_var(self.current_date, self._current_time)
        else:
            return {}
            
    def _get_state(self):
        return (self._current_time, self._last_4_num_shares)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)
        
        # Reset number of current share
        self._current_time = 0
        self._last_4_num_shares = self.ns
                
        # set the current date again => set the orderbook snapshot
        self.current_date = np.random.default_rng().integers(low = 0, high = self.num_date)
        
        # Reset the number of shares and number of time units
        curr_state = self._get_state()

        # get market info and add it into info
        market_var = self.get_market_var()
        
        return curr_state, market_var
        
    def step(self, action):
        # Calculate the reward
        last_num_shares = self._last_4_num_shares % (self.ns + 1)
        reward = self.orderbook.compute_rewards(last_num_shares, self.current_date, self._current_time, action)
        
        # Update the new state
        oldest_num_shares = self._last_4_num_shares // ((self.ns + 1)**3)
        newest_num_shares = last_num_shares - action
        self._last_4_num_shares = (self.ns + 1) * (self._last_4_num_shares - oldest_num_shares * (self.ns + 1) ** 3) + newest_num_shares
        self._current_time += 1
        observation = self._get_state()
        
        # Check if terminated
        terminated = (self._current_time >= self.num_time_units)

        # get market info and add it into info
        market_var = self.get_market_var()
        
        return observation, reward, terminated, False, market_var