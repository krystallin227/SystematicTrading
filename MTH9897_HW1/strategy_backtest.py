import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Strategy:
    def __init__(self, bond_data, percentile=10):
        """
        Initialize the strategy analysis with bond data.
        :param percentile: Top and bottom percentile for long/short strategy.
        """
        bond_data['duration_bucket'] =  pd.cut(bond_data['duration'], [1,  5, 10, 15, 30])
        bond_data.dropna(subset = 'duration_bucket', inplace = True)
        bond_data['ret'] = bond_data.groupby('isin')['dirty_price'].pct_change()
        bond_data['ret_next_m'] = bond_data.groupby('isin')['ret'].shift(-1)
        bond_data.dropna(subset= 'ret_next_m', inplace=True)
        self.bond_data = bond_data
        self.percentile = percentile  # Define the top and bottom percentiles
        self.strategy_returns = None

    def run_strategy(self, weighting):
        """
        Runs the monthly duration-neutral long/short strategy for each company individually.
        """
        # Initialize a list to store strategy returns for each company
        company_returns = []
        bond_data = self.bond_data.copy()
        
        # Group by company and month
        bond_data['month'] = pd.to_datetime(bond_data['date']).dt.to_period('M')
        
        # Calculate the long/short returns            
        for (company, month), group in bond_data.groupby(['company_symbol', 'month']):
        
            if weighting == 'equal' or weighting == 'weighted':
                
                top_cutoff = np.percentile(group['t_spread_resid'], 100 - self.percentile)
                bottom_cutoff = np.percentile(group['t_spread_resid'], self.percentile)

                # Select top (rich) and bottom (cheap) bonds
                rich_bonds = group.query('t_spread_resid >= @top_cutoff').copy()
                cheap_bonds = group.query('t_spread_resid <= @bottom_cutoff').copy()
            
                if weighting == 'equal':
                    rich_bonds['weight'] = 1 / len(rich_bonds)
                    cheap_bonds['weight'] = 1 / len(cheap_bonds)
                
                elif weighting == 'weighted':
                    rich_bonds['weight'] = 1 / (rich_bonds.t_spread_resid.abs() + 0.00001)
                    cheap_bonds['weight'] = 1 / (cheap_bonds.t_spread_resid.abs() + 0.00001)
                
            elif weighting == 'duration_based':
                
                cheap_bonds = pd.DataFrame()
                rich_bonds = pd.DataFrame()
                for _ , df in group.groupby('duration_bucket', observed=False):
                    if df.empty:
                        continue
                    top_cutoff = np.percentile(df['t_spread_resid'], 100 - self.percentile)
                    bottom_cutoff = np.percentile(df['t_spread_resid'], self.percentile)
                    rich_bond = df.query('t_spread_resid >= @top_cutoff').copy()
                    cheap_bond = df.query('t_spread_resid <= @bottom_cutoff').copy()
                    cheap_bonds = pd.concat([cheap_bonds, cheap_bond])
                    rich_bonds = pd.concat([rich_bonds, rich_bond])

                cheap_bonds['weight']  = 1 
                rich_bonds['weight']  = 1 
            
            cheap_bonds['weight'] = cheap_bonds['weight'] / cheap_bonds['weight'].sum() * 10
            # Ensure the strategy is duration-neutral by weighting based on duration
            long_duration = (cheap_bonds['duration'] * cheap_bonds['weight']).sum()
            short_duration =  (rich_bonds['duration'] * rich_bonds['weight']).sum()
            
            if long_duration > 0 and short_duration > 0:
                rich_bonds['weight'] = rich_bonds['weight'] * long_duration / short_duration
                # Calculate the duration-neutral return for long and short positions
                long_return = (cheap_bonds['ret_next_m'] * cheap_bonds['weight'] ).sum()
                short_return = -(rich_bonds['ret_next_m'] * rich_bonds['weight']).sum()
                
                # Net return is the difference, creating a duration-neutral long/short portfolio
                net_return = long_return + short_return
            else:
                net_return = 0  # If no bonds available to trade
            
            # Append the results to company returns
            company_returns.append({
                'company': company,
                'month': month,
                'net_return': net_return
            })
        
        # Convert to DataFrame
        self.strategy_returns = pd.DataFrame(company_returns)
    
    def calculate_performance(self):
        """
        Calculate annualized performance metrics such as Sharpe ratio, turnover, and max drawdown.
        """
        if self.strategy_returns is None:
            raise ValueError("Strategy has not been run yet. Call run_strategy() first.")
        
        # Initialize a list to store performance metrics for each company
        company_metrics = []
        
        # Group bond data by company and calculate performance metrics
        grouped = self.strategy_returns.groupby('company')
        
        for company, group in grouped:
            # Calculate mean returns and volatility for the group
            mean_return = group['net_return'].mean()
            volatility = group['net_return'].std()
            
            # Annualize the mean return and volatility
            annualized_return = mean_return * 12
            annualized_volatility = volatility * np.sqrt(12)
            
            # Calculate Sharpe ratio (risk-adjusted return)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            
            # Turnover calculation: Absolute change in portfolio weights over time
            group['weight_change'] = group['net_return'].diff().abs()  # Change in net returns as a proxy for weight change
            turnover = group['weight_change'].mean()  # Mean absolute weight change as turnover
            
            # Calculate cumulative returns
            group = group.copy()
            group['cumulative_return'] = (1 + group['net_return']).cumprod() - 1
            
            # Calculate rolling maximum of cumulative returns
            group['rolling_max'] = group['cumulative_return'].cummax()
            
            # Calculate drawdown as the difference between the rolling maximum and current cumulative return
            group['drawdown'] = group['rolling_max'] - group['cumulative_return']
            
            # Max drawdown is the largest drawdown during the period
            max_drawdown = group['drawdown'].max()
            
            # Append the metrics for the company
            company_metrics.append({
                'company': company,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'turnover': turnover,
                'max_drawdown': max_drawdown
            })
        
        # Convert the results into a DataFrame and return it
        performance_df = pd.DataFrame(company_metrics)
        return performance_df

    def plot_performance(self):
        """
        Plot cumulative returns and other metrics for the strategy.
        """
        if self.strategy_returns is None:
            raise ValueError("Strategy has not been run yet. Call run_strategy() first.")
        
        # Plot cumulative returns for each company
        grouped = self.strategy_returns.groupby('company')
        plt.figure(figsize=(10, 6))
        
        for company, group in grouped:
            group = group.copy()
            group['month'] = group['month'].dt.to_timestamp()  # Convert Period to timestamp
            group = group.set_index('month').sort_index()
            cumulative_return = (1 + group['net_return']).cumprod() - 1
            plt.plot(cumulative_return.index, cumulative_return, label=company)
        
        plt.title('Cumulative Returns by Company')
        plt.xlabel('Month')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.show()
