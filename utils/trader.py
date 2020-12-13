import os
from utils.quotes import Quotes


class Trader(object):
    def __init__(self, max_hold=10, starting_balance=30000):
        self.quotes = Quotes()
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.positions = {}

        # arbitrary parameters
        self.max_hold = max_hold
        self.trade_fee = 5.00

    def trade_on_signal(self, symbol: str, signal: str, timestamp: int):
        """
        Trades on signal from RL model.
        Buy if bullish. Sell if in position on Bearish.
        Currently does not short on bearish signals.
        Can only hold `self.max_hold` positions.
        """
        if (
            signal == "BULLISH"
            and len(self.positions.keys()) < self.max_hold
            and symbol not in self.positions
        ):
            self.rebalance(symbol, timestamp)

        elif signal == "BEARISH" and symbol in self.positions:
            self.rebalance(symbol, timestamp, True)

    def rebalance(self, symbol: str, timestamp: int, remove=False):
        # trading fee (also deters frequent trades)
        self.balance -= self.trade_fee

        positions = list(self.positions.keys())
        close = lambda s: self.quotes.get_quote(s, timestamp)

        # sell all positions
        for symbol, qty in self.positions.items():
            self.balance += close(symbol) * qty

        if remove:
            positions = [p for p in positions if p != symbol]
        else:
            positions.append(symbol)

        self.positions = {}
        if len(positions) == 0:
            return

        # re-enter positions
        target_val = self.balance / len(positions)
        for symbol in positions:
            qty = target_val // close(symbol)
            self.balance -= qty * close(symbol)
            self.positions[symbol] = qty

    def reward(self, timestamp: int):
        """
        Reward is the current ROI.
        Calculates current value of positions over intial capital.
        """
        positions = self.positions.keys()
        close = lambda s: self.quotes.get_quote(s, timestamp)
        value = (
            sum([close(symbol) * qty for symbol, qty in self.positions.items()])
            + self.balance
        )

        return (value / self.starting_balance - 1) * 100
