"""
Functions for analysing raw data
"""

import glob
import pickle
import functools
import datetime
import collections

import numpy as np

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
    present_value_per_asset = trades_frame_reindexed * token_prices
    present_value = present_value_per_asset.sum(1)

    # calculate PnL
    pnl = present_value.diff()

    # calculate gross market value, return on GMV
    gmv = (trades_frame_reindexed * token_prices).abs().sum(1)
    ret = pnl / gmv

    # how many days was the portfolio active for? A bit of a guess
    n_active = np.isfinite(ret).sum() / 30.0 # in months, approx

    # return
    return dict(
        PV = present_value,
        PV_per_asset = present_value_per_asset,
        positions = trades_frame_reindexed,
        pnl = pnl,
        ret = ret,
        GMV = gmv,
        n_trades = len(trades),
        n_active = n_active,
        )


RANKING_METHODS=[
    'REALISED_RETURN',
    'HIT_RATE',
    'MONTHLY_RETURN',
    'RETURN_PER_TRADE',
    'SHARPE_RATIO',
    'RESIDUAL_RETURN',
    ]


def return_ranking_statistic(performance_stats, ranking_method, token_prices):
    ranking_fn = {
        'REALISED_RETURN'  : rank_realised_vs_unrealised,
        'HIT_RATE'         : rank_hit_rate,
        'MONTHLY_RETURN'   : rank_monthly_return,
        'RETURN_PER_TRADE' : rank_return_per_trade,
        'SHARPE_RATIO'     : rank_sharpe_ratio,
        'RESIDUAL_RETURN'  : rank_residual_return,
        }[ ranking_method ]

    return ranking_fn(performance_stats, token_prices)


def rank_realised_vs_unrealised(performance_stats, token_prices):
    # realised PnL is "cash held", i.e. USDC, USDT, DAI
    # possibly negative
    # unrealised PnL is total PnL - realised PnL
    realised_pnl = performance_stats['PV_per_asset'].loc[
        :, ['USDC', 'USDT', 'DAI']].iloc[-1].sum()

    # it's tricky to compare meaningfully total, realised and unrealised PnL
    # since there are other metrics at play, here we just look at average monthly
    # return of realised PnL, normalised by GMV
    return realised_pnl / performance_stats['GMV'].mean() / performance_stats['n_active']

def rank_hit_rate(performance_stats, token_prices):
    """
    Use FIFO decomposition on individual token position, calculate per-trade PnL
    on individual legs, calculate hit rate. Exclude USDC, USDT, DAI, since they
    have approx constant prices.
    """
    # exclude stablecoins
    pos = performance_stats['positions']
    tokens = pos.columns

    tokens = [tt for tt in tokens if tt not in ['USDC', 'USDT', 'DAI']]
    pos_no_stable = pos.loc[:, tokens]

    fifo_results = get_fifo_trades(
        pos_no_stable, token_prices,
        datetime.datetime(2021, 3, 16))
    trade_rets = np.array([ x[3] for x in fifo_results ])
    return (trade_rets > 0).mean()


def rank_monthly_return(performance_stats, token_prices):
    """Nice and easy, average monthly return on GMV"""
    return performance_stats['ret'].mean() * 30.0

def rank_return_per_trade(performance_stats, token_prices):
    """Same as above, but n_trades as denominator"""
    return performance_stats['ret'].sum() / performance_stats['n_trades']

def rank_sharpe_ratio(performance_stats, token_prices):
    """
    Risk-adjusted return; average annual return / realised annual PnL volatility
    """
    ret = performance_stats['ret']
    # note usually a factor of sqrt(260) is used to annualise, since this is
    # approx number of trading days in a year. But crypto trades every day
    return ret.mean() / ret.std() * np.sqrt(365)

def rank_residual_return(performance_stats, token_prices):
    """
    Compare the return to buy-and-hold BTC
    """
    # identify the first date of trading
    trading_start = performance_stats['ret'].dropna().index[0]

    # make a dummy portfolio holding 1 BTC and short USDC in appropriate amount
    btcusd_start_price = token_prices.loc[trading_start, 'BTC']

    dummy_trade = [
        q.RawTransaction(
            timestamp = trading_start,
            token0 = 'BTC',
            token1 = 'USDC',
            amount0In = 1.0,
            amount1In = 0.0,
            amount0Out = 0.0,
            amount1Out = btcusd_start_price,
            amountUSD = np.nan,
            to = 'BAD_WALLET',
            sender = 'BAD_WALLET',
            )
        ]

    # calculate its performance stats, in particular return
    dummy_trade_stats = get_wallet_performance_stats(dummy_trade, token_prices)

    # calculate how much excess return the given portfolio produced
    excess_return = (
        performance_stats['ret'].mean()
        - dummy_trade_stats['ret'].mean()) * 30.0

    return excess_return


def datetime_to_fractional_year(datetime_):
    benchmark_time = datetime.datetime(datetime_.year, 1, 1)
    difference_days = (datetime_ - benchmark_time).total_seconds() / 86400.0

    return datetime_.year + difference_days / 365.25


class FifoTradeQueue:
    """Maintaing a queue of orders for FIFO decomposition"""
    def __init__(self, max_size=10000):
        self.queue = collections.deque(maxlen=max_size)

    def push(self, timestamp, trade):
        """
        Add new trade; if can emit some closed trades out of that, then do so.
        """
        # if queue empty, just pop it back
        if len(self.queue) == 0:
            self.queue.append([timestamp, trade])
            return []
        else:
            emitted_trades = []
            # while the leftmost trade and the new one being added
            # point in opposite direction
            remaining_trade = trade
            while ((len(self.queue) > 0)
                   and (self.queue[0][1] * trade < 0)
                   and (not np.isclose(remaining_trade, 0.0))):
                # if remaining_trade is less than the leftmost trade, take that
                # and reduce leftmost trade, set remaining_trade to 0
                leftmost_timestamp, leftmost_trade = self.queue[0]
                if np.abs(leftmost_trade) > np.abs(trade):
                    emitted_trades.append((
                        -remaining_trade, # - because this is the closeout
                        leftmost_timestamp,
                        timestamp))
                    self.queue[0][1] += remaining_trade
                    remaining_trade = 0.0
                else:
                    # else pop the leftmost trade, subtract from remaining_trade
                    # and continue
                   emitted_trades.append((
                       leftmost_trade,
                       leftmost_timestamp,
                       timestamp))
                   self.queue.popleft()
                   remaining_trade += leftmost_trade
            # if there is anything remaning in remaining_trade, pop it on the
            # right side of the queue
            if not np.isclose(remaining_trade, 0.0):
                self.queue.append([timestamp, remaining_trade])

            return emitted_trades

def get_fifo_trades(positions, prices, final_unwind_timestamp=None):
    """
    For each position sequence, return a tuple (token, t1, t2, return)
    """

    # helper for below
    def get_price(token, timestamp):
        # return prices.reindex(index=[timestamp], columns=[token],
        #                       method='ffill').values[0,0]
        return prices[token].reindex(index=[timestamp], method='ffill').values[0]

    result = []
    for token in positions.columns:
        token_pos = positions.loc[:, token]
        if final_unwind_timestamp is not None:
            token_pos = token_pos.copy()
            token_pos[final_unwind_timestamp] = 0.0

        token_trades = token_pos.dropna().diff().dropna()
        token_trades = token_trades[token_trades != 0]

        timestamps = token_trades.index
        trades = token_trades.values

        queue = FifoTradeQueue()

        for timestamp, trade in zip(timestamps, trades):
            popped_trades = queue.push(timestamp, trade)
            for amount, t1, t2 in popped_trades:
                # get token ret
                # p1, p2 = prices.loc[t1, token], prices.loc[t2, token]
                p1 = get_price(token, t1)
                p2 = get_price(token, t2)
                token_ret = p2 / p1 - 1.0
                result.append((token, t1, t2, token_ret))

        return result
