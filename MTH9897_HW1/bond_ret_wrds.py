import wrds
import os

db = wrds.Connection()

def bond_data(ticker):

    query = f"""
    SELECT date, isin, company_symbol, bond_type, security_level, conv, rating_cat, day_count_basis, first_interest_date, maturity, treasury_maturity,tmt, t_dvolume,t_volume, ncoups, t_spread, coupon, coupamt, coupacc, price_ldm, duration, amount_outstanding, offering_amt
    FROM wrdsapps.bondret
    WHERE company_symbol IN ('{ticker}') 
    AND date >= '2017-01-01' AND date <= '2022-09-30'
    ORDER BY date;
    """

    #fetch the data 
    df_bond_data = db.raw_sql(query)
    if df_bond_data.empty: return 
    
    most_freq_rating = df_bond_data['rating_cat'].mode()[0]
    if most_freq_rating not in ["A", "AA", "AAA"]:
        return
    
    #filter data for the same bond type and security level
    df_bond_data = df_bond_data.query('bond_type == "CDEB" and security_level == "SEN" and rating_cat == @most_freq_rating and conv == 0')
        
    if not os.path.exists('HW1/data'):
        os.makedirs('HW1/data')
        
    data_path = f'HW1/data/{ticker}.csv'
    df_bond_data.to_csv(data_path, index=False)
    return df_bond_data

if __name__ == '__main__':

    for ticker in ['AAPL', 'AMZN', 'IBM', 'MSFT', 'WMT']:
        bond_data(ticker)