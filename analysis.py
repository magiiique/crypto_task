"""
Functions for analysing raw data
"""

import glob
import pickle
import functools
import datetime

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
    # FIXME
    return 0.0

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


#def plot_wallet_
