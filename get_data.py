import queries
import pickle
import itertools
import tqdm

OUTPUT='scraped.pickle'

def main():
    wallets = list(itertools.islice(
        queries.get_wallets(), 0, 100))

    all_transactions = [
        queries.get_user_transactions_v2(wallet)
        for wallet in tqdm.tqdm(wallets)]

    data = dict(zip(wallets, all_transactions))

    with open(OUTPUT, 'wb') as ff:
        pickle.dump(data, ff)

if __name__ == '__main__':
    main()
