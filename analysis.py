"""
Functions for analysing raw data
"""

import glob
import pickle
import functools

import queries as q
import bitfinex_data as bd

def filter_trades(trades, covered_tokens):
    """
    Filters a list of trades to one in fully covered assets
    """
    return [
        trade for trade in trades
        if trade.token0 in covered_tokens
        and trade.token1 in covered_tokens
        ]

def get_wallets(paths, filter_tokens=True):
    """
    Loads tokens, by default filtering them down to ones with Bitfinex data. To
    do no filtering, specify filter_tokens=None, or to do custom filtering,
    specify a list of tokens to filter on.
    """
    total = dict()
    for path in glob.glob(paths):
        with open(path, 'rb') as ff:
            wallets = pickle.load(ff)
            total.update(wallets)

    if filter_tokens == True:
        filter_tokens = bd.COVERED_TOKENS

    if filter_tokens is not None:
        total = {
            wallet : filter_trades(trades, filter_tokens)
            for wallet, trades in total.items()
            }

    return total

def filter_wallets(wallets, min_date_span=180, min_tokens=5, min_trades=20):
    """
    Filters wallets according to simple criteria:
    - min_date_span - min length of time the wallet was active
      (first to last trade)
    - min_tokens - min number of tokens traded (excluding USDC/USDT - that's
      just numeraire)
    - min_trades - min number of trades executed

    Specify None for any criterion to not filter on it
    """
    def check_date_span(trades, date_span):
        if len(trades) == 0:
            return False
        else:
            # span in days
            span = (trades[-1].timestamp - trades[0].timestamp).total_seconds() / 86400.0
            return (span >= date_span)

    def check_min_tokens(trades, n_tokens):
        """Doesn't count USDC/USDT"""
        token_set = functools.reduce(
            set.union,
            [{trade.token0, trade.token1}
             for trade in trades])

        # exclude USDC, USDT in the count
        # token_set -= {'USDC', 'USDT'}

        return (len(token_set) >= n_tokens)

    def check_min_trades(trades, n_trades):
        # simple len check
        return len(trades) >= n_trades


    def trades_comply(trades):
        for predicate, value in [(check_date_span, min_date_span),
                                 (check_min_tokens, min_tokens),
                                 (check_min_trades, min_trades),
                                 ]:
            if value is None:
                continue
            if not predicate(trades, value):
                return False

        return True

    return {wallet : trades
            for wallet, trades in wallets.items()
            if trades_comply(trades)}


def get_wallet_performance_stats(trades, token_prices):
    """Returns wallet value, daily PnL, on the subset of priced tokens"""
    # put trades into positions
    trades_frame = q.trades_to_positions(trades)

    # push them to date boundaries
    trades_frame_reindexed = trades_frame.reindex(
        index = token_prices.index,
        method='ffill')

    # calculate PV
    present_value = (trades_frame_reindexed * token_prices).sum(1)

    # calculate PnL
    pnl = present_value.diff()

    # calculate gross market value, return on GMV
    gmv = (trades_frame_reindexed * token_prices).abs().sum(1)
    ret = pnl / gmv

    # return
    return dict(
        PV = present_value,
        pnl = pnl,
        ret = ret,
        GMV = gmv,
        n_trades = len(trades),
        )
