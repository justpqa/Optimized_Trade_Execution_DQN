import numpy as np
import pandas as pd

# Custom class for the order book environment with calculation for market impact and slippage
class OrderBookEnv:
    def __init__(self, data: pd.DataFrame, has_market_var: bool = False):
        self.data = data
        self.has_market_var = has_market_var
    
    def calculate_vwap(self, date_idx, idx, shares):
        """
        Calculates the Volume-Weighted Average Price (VWAP) for a given step and share size.

        Parameters:
        idx (int): The index of the current step in the market data.
        shares (int): The number of shares being traded at the current step.

        Returns:
        float: The calculated VWAP price for the current step.
        """
        # Assumes you have best 5 bid prices and sizes in your dataset
        bid_prices = [self.data.loc[390*date_idx + idx, f'bid_price_{i}'] for i in range(1,6)]
        bid_sizes = [self.data.loc[390*date_idx + idx, f'bid_size_{i}'] for i in range(1,6)]
        cumsum = 0
        for i, size in enumerate(bid_sizes):
            cumsum += size
            if cumsum >= shares:
                break
        
        return np.sum(np.array(bid_prices[:i+1]) * np.array(bid_sizes[:i+1])) / np.sum(bid_sizes[:i+1])
    
    def compute_rewards(self, curr_num_shares, date_idx, idx, shares, alpha = 4.439584265535017e-06):
        """
        Computes the reward based on components such as slippage and market impact for a given trade.

        Parameters:
        alpha (float): A scaling factor for market impact (determined empirically or based on research).
        shares (int): The number of shares being traded at the current step.
        idx (int): The index of the current step in the market data.

        Returns:
        float: A number represent the reward, -inf if we do not have enough shares to solde
        """
        if shares > curr_num_shares:
            return -10**9
        if idx == 389 and curr_num_shares > shares:
            return -10**9
        actual_price = self.calculate_vwap(date_idx, idx, shares)
        Slippage = (self.data.loc[390*date_idx + idx, 'bid_price_1'] - actual_price) * shares  # Assumes bid_price is in your dataset
        Market_Impact = alpha * np.sqrt(shares)
        return Slippage - Market_Impact

    def compute_market_var(self, date_idx: int, idx: int):
        if self.has_market_var:
            if idx < 390:
                return {
                    "bid-ask spread": self.data.loc[390*date_idx + idx, 'bid_price_1'] - self.data.loc[390*date_idx + idx, 'ask_price_1'],
                    "volume imbalance": self.data.loc[390*date_idx + idx, 'bid_size_1'] / (self.data.loc[390*date_idx + idx, 'bid_size_1'] + self.data.loc[390*date_idx + idx, 'ask_size_1'])
                }
            else:
                return {
                    "bid_ask spread": 0,
                    "volume imbalance": 0
                }
        else:
            return {}