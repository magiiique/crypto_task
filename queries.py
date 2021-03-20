"""
Queries to The Graph db
"""

import json
import requests
import logging
import itertools
import collections
import datetime
import time
from functools import reduce
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

THE_GRAPH_URL="https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"

# make some standard queries
USERS_QUERY="""
{{
  users(first: {first}, skip: {skip}) {{
    id
  }}
}}
"""

SWAPS_QUERY="""
{{
  swaps(where: {{to: "{user_id}"}}, orderBy: timestamp, first: 1000)
    {{
      timestamp

      pair {{
        token0 {{symbol}}
        token1 {{symbol}}
      }}

      amount0In
      amount1In
      amount0Out
      amount1Out
      to
      sender
      amountUSD
    }}
}}
"""

SWAPS_QUERY_V2="""
{{
  swaps(where: {{to: "{user_id}", timestamp_gt: "{last_timestamp}"}},
        orderBy: timestamp, first: 1000)
    {{
      timestamp

      pair {{
        token0 {{symbol}}
        token1 {{symbol}}
      }}

      amount0In
      amount1In
      amount0Out
      amount1Out
      to
      sender
      amountUSD
    }}
}}
"""

SAMPLE_QUERY="""
{
  uniswapFactories(first: 5) {
    id
    pairCount
    totalVolumeUSD
  }
  tokens(first: 5) {
    id
    symbol
    name
    decimals
  }
}
"""

# add some basic rate limiting
def rate_limit(rate_per_sec):
    delay = 1/rate_per_sec
    def inner(fn):
        last_time = None
        def wrapped(*args, **kwargs):
            nonlocal last_time
            now = time.time()
            if last_time is not None:
                elapsed = now-last_time
                if elapsed < delay:
                    time.sleep(delay-elapsed)
            last_time = now
            return fn(*args, **kwargs)
        return wrapped
    return inner

_SESSION=None
@rate_limit(0.5)
def get_query_result(query, url=None, variables=None):
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    if url is None:
        url = THE_GRAPH_URL

    headers = {'Content-type' : 'application/json'}
    data = {'query': query}
    if variables is not None:
        data['variables'] = variables
    # response = requests.post(
    response = _SESSION.post(
        url,
        data=json.dumps(data),
        headers=headers)
    return response.text


def get_wallets(batch_size=1000):
    """Iterator returning wallets from the users table"""
    skip = 0
    while True:
        query = USERS_QUERY.format(
            first=batch_size,
            skip=skip)
        result = json.loads(get_query_result(query))
        if 'errors' in result:
            raise ValueError("Error executing query")
        else:
            ids = result['data']['users']
            for id_ in ids:
                yield id_['id']


RawTransaction = collections.namedtuple(
    'RawTransaction',
    ['timestamp',
     'token0',
     'token1',
     'amount0In',
     'amount1In',
     'amount0Out',
     'amount1Out',
     'amountUSD',
     'to',
     'sender'
     ])

def get_user_transactions(user_id):
    transactions = []
    query = SWAPS_QUERY.format(user_id=user_id)
    result = json.loads(get_query_result(query))
    if 'errors' in result:
        pass
    else:
        for swap in result['data']['swaps']:
            transactions.append( RawTransaction(
                timestamp = datetime.datetime.fromtimestamp(int(swap['timestamp'])),
                token0 = swap['pair']['token0']['symbol'],
                token1 = swap['pair']['token1']['symbol'],
                amount0In = float(swap['amount0In']),
                amount1In = float(swap['amount1In']),
                amount0Out = float(swap['amount0Out']),
                amount1Out = float(swap['amount1Out']),
                amountUSD = float(swap['amountUSD']),
                to = swap['to'],
                sender = swap['sender'],
                ) )
    return transactions


def get_user_transactions_v2(user_id):
    last_timestamp=0
    transactions = []
    can_continue = True
    while can_continue:
        query = SWAPS_QUERY_V2.format(
            user_id=user_id,
            last_timestamp=last_timestamp)
        result = json.loads(get_query_result(query))
        if 'errors' in result:
            can_continue = False
        elif len(result['data']['swaps']) == 0:
            can_continue = False
        else:
            for swap in result['data']['swaps']:
                transactions.append( RawTransaction(
                    timestamp = datetime.datetime.fromtimestamp(int(swap['timestamp'])),
                    token0 = swap['pair']['token0']['symbol'],
                    token1 = swap['pair']['token1']['symbol'],
                    amount0In = float(swap['amount0In']),
                    amount1In = float(swap['amount1In']),
                    amount0Out = float(swap['amount0Out']),
                    amount1Out = float(swap['amount1Out']),
                    amountUSD = float(swap['amountUSD']),
                    to = swap['to'],
                    sender = swap['sender'],
                    ) )
            last_timestamp = int(swap['timestamp'])
    return transactions


def main():
    # print( get_query_result(SAMPLE_QUERY) )
    wallets = list(itertools.islice(
        get_wallets(), 0, 10))

    all_transactions = [
        # get_user_transactions(wallet)
        get_user_transactions_v2(wallet)
        for wallet in wallets]

    for wallet, tr in zip(wallets, all_transactions):
        ll = len(tr)
        if ll > 0:
            span = (tr[-1].timestamp - tr[0].timestamp).total_seconds() / 84600.0
            # how many active contracts?
            contract_set = reduce(
                set.union,
                ({rt.token0, rt.token1}
                 for rt in tr))
            ncontracts = len(contract_set)
        else:
            span = 0
            ncontracts = 0

        print(wallet, ll, span, ncontracts)


def trades_to_positions(trades):
    """
    Returns a pandas df with snapshots of position held. Assumption is that
    portfolio is initially empty; even though in practice it must be seeded with
    some crypto for exchanges to happen. But this is good enough for assessing
    performance.
    """
    # get unique currencies
    unique_ccys_set = reduce(
        set.union,
        [{trade.token0, trade.token1}
         for trade in trades])
    unique_ccys = sorted(unique_ccys_set)

    # build up to a frame
    positions = {token: 0 for token in unique_ccys}
    timestamps = []
    values = []

    for trade in trades:
        timestamps.append(trade.timestamp)
        # go through the trades
        new_positions = positions.copy()
        # again, not documented, I'm guessing 'amount in' is amount going into
        # the exchange, i.e. out of the wallet, and vice versa
        for token, amount in [
                (trade.token0, -trade.amount0In),
                (trade.token0, +trade.amount0Out),
                (trade.token1, -trade.amount1In),
                (trade.token1, +trade.amount1Out),
                ]:
            new_positions[token] = new_positions[token] + amount
        values.append(
            np.array([ new_positions[token] for token in unique_ccys ]) )
        positions = new_positions

    values = np.vstack(values)

    # return
    return pd.DataFrame(
        values,
        index = timestamps,
        columns = unique_ccys,
        )

OFFLINE_TOKENS = {
    "BTC",
    "ETH",
    "LTC",
    "AID",
    "BAT",
    "LTC",
    "DAI",
    "DASH",
    "EDO",
    "EOS",
    "ETC",
    "ETP",
    "NEO",
    "OMG",
    "QTUM",
    "REP",
    "TRX",
    "XLM",
    "XMR",
    "XVG",
    # just to make my life easier
    "USDT",
    "USDC",
    }


if __name__ == '__main__':
    main()
