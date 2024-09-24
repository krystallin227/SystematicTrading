import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

class SpreadCurve:
    
    def __init__(self, tickers, features, target_var, x_var) -> None:
        self.tickers = tickers
        self.features = features
        self.x_var = x_var
        self.target_var = target_var
        self.bond_data = self.get_bond_data(tickers, features + [target_var, 'frac_outstanding', 'dirty_price', 'duration'])
        self.fitted_models = None
    

    def get_bond_data(self, tickers, dropna_cols):
        bond_data = pd.concat([pd.read_csv(f'data/{ticker}.csv', parse_dates= ['date', 'first_interest_date']) for ticker in tickers])
        bond_data['age'] =  ((bond_data['date'] - bond_data['first_interest_date']).dt.days / 365).clip(lower = 0)
        bond_data[f'log_{self.x_var}'] = np.log(1 + bond_data[self.x_var])
        bond_data.sort_values(by = ['date', 'company_symbol', self.x_var], inplace = True)
        bond_data['t_spread']  = bond_data['t_spread'] * 10000 #convert t_spread to bps
        bond_data.query(f'{self.x_var} > 0.5', inplace=True)
        bond_data.eval('frac_outstanding = amount_outstanding / offering_amt', inplace=True)
        bond_data['nearst_pillar'] = bond_data[self.x_var].round()
        bond_data.eval('dirty_price = price_ldm + coupacc', inplace = True)        
        bond_data.dropna(subset= dropna_cols, inplace = True)
        bond_data.reset_index(drop=True, inplace=True)
        return bond_data
    
    def t_spread_plots(self, query_date=None):
        fig, axs = plt.subplots(3, 2, figsize=(9, 7.5))
        fig.suptitle(f'T Spread vs Duration {query_date.strftime("%Y-%m-%d") if query_date else ""}', fontsize=20)
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, ticker in enumerate(self.tickers):
            ax = axs[i // 2, i % 2]
            
            if query_date:
                subset = self.bond_data.query('date == @query_date and company_symbol == @ticker')
            else:
                subset = self.bond_data.query('company_symbol == @ticker')
            
            # Define the size of the bubbles based on 't_spread'
            sizes = subset['amount_outstanding'] / 1e5   # Adjust the multiplier as necessary for better scaling
            
            # Scatter plot with bubble size based on 't_spread'
            scatter = ax.scatter(subset['duration'], subset['t_spread'], s=sizes, label=ticker, color=colors[i % len(colors)], alpha=0.6, edgecolor='black')
            
            ax.set_title(f'{ticker}')
            ax.set_xlabel('Duration')
            ax.set_ylabel('t_spread (bp)')
        
        axs[-1][-1].axis('off')
        
        plt.tight_layout()
        plt.show()

    # Define a function to run the regression on each group
    def _run_regression(self, group, weight_col = None):
        X = group[self.features]
        y = group[self.target_var]
        
        # Add a constant (intercept) to the model
        X = sm.add_constant(X)
        
        # Fit the regression model
        if weight_col:
            model = sm.WLS(y, X, weights=group[weight_col]).fit()
        else:
            model = sm.OLS(y, X).fit()
        
        return model

    def fit(self, weight_col):
        
        #remove pillars very close to each other
        dfs = []
        for _, df in self.bond_data.groupby(['date', 'company_symbol']):
            dfs.append(df.sort_values(['nearst_pillar', weight_col]).drop_duplicates(subset=['nearst_pillar'], keep='last'))

        self.training_data = pd.concat(dfs)
        self.fitted_models = self.training_data.groupby(['date', 'company_symbol']).apply(lambda x: self._run_regression(x, weight_col))
        
        for (date, company), df in self.bond_data.groupby(['date', 'company_symbol']):
            model = self.fitted_models[date][company]
            X = sm.add_constant(df[self.features])
            self.bond_data.loc[df.index, 't_spread_fitted'] = model.predict(X)
            
        
        
    def plot_fitted_vs_actual(self, query_date):
        assert self.fitted_models is not None, 'Must call fit before plotting'
        
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(3, 2, figsize=(9, 7.5))
        fig.suptitle(f'Fitted vs Actual T Spread for {query_date.strftime("%Y-%m-%d")}', fontsize=20)
        
        # Get the list of unique company symbols
        companies = self.bond_data['company_symbol'].unique()
        
        for i, company in enumerate(companies):
            ax = axs[i // 2, i % 2]
            
            # Filter bond data and fitted models for the specific company and query_date
            subset = self.bond_data.query('date == @query_date and company_symbol == @company')
            subset_train = self.training_data.query('date == @query_date and company_symbol == @company')
                        
            if subset.empty:
                ax.set_title(f'{company}')
                ax.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue
            
            model = self.fitted_models[query_date][company]
            # Get actual t_spread and fitted values
            fitted_t_spread = model.fittedvalues
            actual_t_spread = subset['t_spread']
            
            # Plot actual values (scatter without label)
            ax.plot(subset[self.x_var], actual_t_spread, 'o', alpha=0.7)

            # Plot fitted values as a line (without label)
            ax.plot(subset_train[self.x_var], fitted_t_spread, '^-')

            # Add text labels for coupon values at each fitted point, alternating positions
            for j in range(len(subset_train)):
                tmt_val = subset_train[self.x_var].iloc[j]
                fitted_val = fitted_t_spread.iloc[j]
                coupon_val = subset_train['coupon'].iloc[j]
                
                # Alternate label position
                if j % 2 == 0:
                    ax.text(tmt_val, fitted_val, f'{coupon_val:.2f}', fontsize=8, 
                            verticalalignment='bottom', horizontalalignment='right')
                else:
                    ax.text(tmt_val, fitted_val, f'{coupon_val:.2f}', fontsize=8, 
                            verticalalignment='top', horizontalalignment='left')

            # Customize plot appearance
            ax.set_title(f'{company}')
            ax.set_xlabel(self.x_var)
            ax.set_ylabel('T Spread (bp)')
        
        axs[-1][-1].axis('off')

        plt.tight_layout()
        plt.show()
