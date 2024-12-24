from typing import Any
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from .dqn_agent import DQNAgent
from .trading_env import TradingEnv

def numeric_state_to_image_state(n_state_arr: list[tuple[int, int]], num_shares: int, num_time_units: int):
    # Convert a state that represent last 4 number of stocks => image 
    i_state_arr = []
    for n_state in n_state_arr:
        current_time, last_4_num_stocks = n_state
        i_state = np.zeros((4, num_time_units + 1, num_shares + 1))
        for i in range(4):
            if i <= current_time:
                curr_num_stocks = last_4_num_stocks % (num_shares + 1)
                i_state[i, current_time - i, curr_num_stocks] = 1
                last_4_num_stocks //= (num_shares + 1)
        i_state_arr.append(i_state)
    # shape (len(n_state_arr), 4, num_time_units, num_shares)
    return np.stack(tuple(i_state_arr), axis = 0)

def resize_image_state(batch_big_i_state):
    batch_big_i_state = torch.tensor(batch_big_i_state, dtype=torch.float32)
    resize = transforms.Resize((84, 84))
    batch_big_i_state_resize = torch.stack([resize(batch_big_i_state[i]) for i in range(len(batch_big_i_state))])
    return batch_big_i_state_resize

# Function for training given a dqn agent and a training env
def create_and_train_dqn(agent_params: dict[str, Any], env: TradingEnv, num_episodes: int = 30, batch_size: int = 128, target_update_freq: int = 10):
    # make sure agent know if it includes market var
    agent_params["num_time_units"] = env.num_time_units
    agent_params["num_shares"] = env.ns
    agent_params["has_market_var"] = env.has_market_var
    # Create agent given parameters
    agent = DQNAgent(**agent_params)
    
    # Proceed to training
    for episode in range(num_episodes):
        state, market_var = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, market_var)
            next_state, reward, done, truncated, next_market_var = env.step(action)
            agent.replay_buffer.add((state, market_var, action, reward, next_state, next_market_var, done))

            agent.train(batch_size)

            state = next_state
            market_var = next_market_var
            episode_reward += reward

        
        # Update the target network every few episodes
        if episode % target_update_freq == 0:
            agent.update_target_network()
    
    return agent

# Try to make it choose action to do on the last day
def get_best_actions(agent: DQNAgent, env: TradingEnv, date_inx: int):
    state = env._get_state()
    market_var = env.get_market_var()
    done = False
    best_actions = []
    total_rewards = 0
    while not done:
        action = agent.select_best_action(state, market_var)
        next_state, _, done, _, next_market_var = env.step(action)
        best_actions.append(action)
        curr_num_shares = state[1] % (env.ns + 1)
        total_rewards += env.orderbook.compute_rewards(curr_num_shares, env.current_date, state[0], action)
        state = next_state
        market_var = next_market_var
    best_actions_df = pd.DataFrame({
        "timestamp": env.orderbook.data.iloc[390*date_inx:390*date_inx + 390, 0], 
        "shares": best_actions
    })
    return best_actions_df, total_rewards