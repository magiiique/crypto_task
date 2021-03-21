"""
Script for running the queries against The Graph Uniswap API.
"""

import queries
import pickle
import itertools
import tqdm

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

OUTPUT='scraped.pickle'

def main():
    wallets = list(itertools.islice(
        queries.get_wallets(), 0, 600))

    all_transactions = [
        queries.get_user_transactions_v2(wallet)
        for wallet in tqdm.tqdm(wallets)]

    data = dict(zip(wallets, all_transactions))

    with open(OUTPUT, 'wb') as ff:
        pickle.dump(data, ff)

if __name__ == '__main__':
    main()
