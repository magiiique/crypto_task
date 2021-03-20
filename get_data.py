import queries
import pickle
import itertools
import tqdm
import glob
import collections
from alphavantage_queries import get_currency_data
import pandas as pd
import os

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

OUTPUT='scraped3.pickle'
OUTPUT_PRICES='prices.csv'

def get_fx():
    # get scraped wallets
    wallets_files = glob.glob('scraped*.pickle')

    # get traded currencies
    ccys = []
    for wallets_file in wallets_files:
        with open(wallets_file, 'rb') as ff:
            wallet_trades = pickle.load(ff)
        for wallet, trades in wallet_trades.items():
            for trade in trades:
                ccys.append(trade.token0)
                ccys.append(trade.token1)

    unique, _ = zip(*collections.Counter(ccys).most_common())
    unique = unique[:20]

    for symbol in tqdm.tqdm(unique):
        save_path = os.path.join('exchange_rates', f'{symbol}.csv')
        if os.path.exists(save_path):
            logging.info(f"Skipping token {symbol}, already downloaded")
            continue
        try:
            data = get_currency_data(symbol)
            if data is None:
                logging.warn(f'No data for symbol {symbol}')
            else:
                save_path = os.path.join('exchange_rates', f'{symbol}.csv')
                data.to_csv(save_path)
        except Exception as ee:
            logging.error(f'Exception reading token {symbol}: {ee}')


def main():
    wallets = list(itertools.islice(
        queries.get_wallets(), 300, 600))

    all_transactions = [
        queries.get_user_transactions_v2(wallet)
        for wallet in tqdm.tqdm(wallets)]

    data = dict(zip(wallets, all_transactions))

    with open(OUTPUT, 'wb') as ff:
        pickle.dump(data, ff)

if __name__ == '__main__':
    # main()
    get_fx()
