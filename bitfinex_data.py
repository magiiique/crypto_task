"""
Functions for loading the Bitfinex data from csv
"""

import pandas as pd
import os

COVERED_TOKENS = [
    'BTC', 'WBTC',
    'ETH', 'WETH',
    'LTC',
    'AID',
    'BAT',
    'DAI',
    'DASH',
    'EDO',
    'EOS',
    'ETC',
    'ETP',
    'NEO',
    'OMG',
    'QTUM',
    'REP',
    'TRX',
    'XLM',
    'XMR',
    'XVG',
    'USDC', 'USDT',
    ]

def load_one_token(token):
    if token == 'WETH':
        token = 'ETH'
    elif token == 'WBTC':
        token = 'BTC'

    if token in ['USDC', 'USDT']:
        dates = pd.date_range('2018-01-01', '2021-03-31')
        return pd.Series(1.0, dates)

    path = os.path.join(f'bitfinex_rates/Bitfinex_{token}USD_d.csv')
    if not os.path.exists(path):
        # return a frame of nan
        # return pd.DataFrame(index=pd.DatetimeIndex([]))
        return pd.Series()
    full_frame = pd.read_csv(path, index_col='date', skiprows=1, parse_dates=True)
    close_price = full_frame['close']
    # looks like AID, REP have some innocuous duplicate index values, drop them
    # for now
    close_price_dropped = close_price[~close_price.index.duplicated(keep='last')]
    return close_price_dropped


def load_token_prices(tokens):
    """Returns USD prices of tokens"""
    data = {token : load_one_token(token)
            for token in tokens}
    return pd.DataFrame(data)
