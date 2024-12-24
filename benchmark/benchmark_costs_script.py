import pandas as pd
import numpy as np

class Benchmark:
    
    def __init__(self, data):
        """
        Initializes the Benchmark class with provided market data.

        Parameters:
        data (DataFrame): A DataFrame containing market data, including top 5 bid prices and sizes. (Use bid_ask_ohlcv_data), will have 390 * k rows
        """
        # Use data with top 5 bid prices and sizes for benchmarking
        # make volume
        data["volume"] = 0
        data = data.fillna(0)
        for i in range(1, 6):
            data["volume"] = data["volume"] + data[f"bid_price_{i}"] + data[f"ask_price_{i}"]
        
        # Filter all valid dates with 390 minutes
        data["date"] = data["timestamp"].apply(lambda x: x.split(" ")[0])
        num_minutes_each_date = data.groupby("date").count()["timestamp"]
        valid_date = num_minutes_each_date[num_minutes_each_date == 390].index.tolist()
        data = data[data["date"].isin(valid_date)].reset_index().drop(["date", "index"], axis = 1)
        self.data = data

    def get_twap_trades(self, date_inx, initial_inventory, preferred_timeframe=390):
        """
        Generates a trade schedule based on the Time-Weighted Average Price (TWAP) strategy.

        Parameters:
        date_inx (int): The index of the date the we will use
        initial_inventory (int): The total number of shares to be sold over the preferred timeframe.
        preferred_timeframe (int): The total number of time steps (default is 390, representing a full trading day).

        Returns:
        DataFrame: A DataFrame containing the TWAP trades with timestamps, price, shares sold, and remaining inventory.
        """
        data = self.data.loc[390*(date_inx): 390*(date_inx + 1), :]
        total_steps = len(data)
        twap_shares_per_step = initial_inventory / preferred_timeframe
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            size_of_slice = min(twap_shares_per_step, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'shares': size_of_slice,
            }
            trades.append(trade)
        return pd.DataFrame(trades)
    
    def get_vwap_trades(self, date_inx, initial_inventory, preferred_timeframe=390):
        """
        Generates a trade schedule based on the Volume-Weighted Average Price (VWAP) strategy.

        Parameters:
        date_inx (int): The index of the date the we will use
        initial_inventory (int): The total number of shares to be sold over the preferred timeframe.
        preferred_timeframe (int): The total number of time steps (default is 390, representing a full trading day).

        Returns:
        DataFrame: A DataFrame containing the VWAP trades with timestamps, price, shares sold, and remaining inventory.
        """
        data = self.data.loc[390*(date_inx): 390*(date_inx + 1), :]
        total_volume = data['volume'].sum()
        total_steps = len(data)
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            volume_at_step = data['volume'].iloc[step]
            size_of_slice = (volume_at_step / total_volume) * initial_inventory
            size_of_slice = min(size_of_slice, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'shares': size_of_slice
            }
            trades.append(trade)
        return pd.DataFrame(trades)
    
    def calculate_vwap(self, idx, shares, date_inx):
        """
        Calculates the Volume-Weighted Average Price (VWAP) for a given step and share size.

        Parameters:
        idx (int): The index of the current step in the market data.
        shares (int): The number of shares being traded at the current step.
        date_inx (int): The index of the date the we will use

        Returns:
        float: The calculated VWAP price for the current step.
        """
        # Assumes you have best 5 bid prices and sizes in your dataset
        bid_prices = [self.data.loc[390 * date_inx + idx, f'bid_price_{i}'] for i in range(1,6)]
        bid_sizes = [self.data.loc[390 * date_inx + idx, f'bid_size_{i}'] for i in range(1,6)]
        cumsum = 0
        for i, size in enumerate(bid_sizes):
            cumsum += size
            if cumsum >= shares:
                break
        
        return np.sum(np.array(bid_prices[:i+1]) * np.array(bid_sizes[:i+1])) / np.sum(bid_sizes[:i+1])

    def compute_components(self, alpha, shares, idx, date_inx):
        """
        Computes the transaction cost components such as slippage and market impact for a given trade.

        Parameters:
        alpha (float): A scaling factor for market impact (determined empirically or based on research).
        shares (int): The number of shares being traded at the current step.
        idx (int): The index of the current step in the market data.
        date_inx (int): The index of the date the we will use

        Returns:
        array: A NumPy array containing the slippage and market impact for the given trade.
        """
        actual_price = self.calculate_vwap(idx, shares, date_inx)
        Slippage = (self.data.loc[390 * date_inx + idx, 'bid_price_1'] - actual_price) * shares 
        Market_Impact = alpha * np.sqrt(shares)
        return np.array([Slippage, Market_Impact])
    
    def simulate_strategy(self, trades, date_inx):
        """
        Simulates a trading strategy and calculates various transaction cost components.

        Parameters:
        trades (DataFrame): A DataFrame where each row contains 'shares' and 'action' for each trade.
        data (DataFrame): Market data including bid prices and volumes.
        preferred_timeframe (int): The total number of time steps over which the strategy is simulated.

        Returns:
        tuple: A tuple containing lists of slippage, market impact.
        """
        
        # Initialize result lists
        alpha = 4.439584265535017e-06 
        slippage = []
        market_impact = []
        rewards = []
        shares_traded = []
        # Simulate the strategy
        for idx in range(len(trades)):
            shares = trades.iloc[idx]['shares']
            reward = self.compute_components(alpha, shares, idx, date_inx)
            slippage.append(reward[0])
            market_impact.append(reward[1])
            shares_traded.append(shares)
            rewards.append(reward)
            
        return slippage, market_impact