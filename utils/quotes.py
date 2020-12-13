from polygon import RESTClient
from datetime import datetime
import pickle
import arrow
import os
import json


class Quotes(object):
    def __init__(self):
        if not os.path.exists("price_cache/cache.json"):
            raise ValueError("Missing price cache")

        with open("price_cache/cache.pkl", "rb") as f:
            self.cache = pickle.load(f)

    def get_quote(self, symbol, timestamp):
        return self.cache[f"{symbol}{timestamp}"]

    def __getitem__(self, key):
        return self.cache[key]


class QuotesFetcher(object):
    def __init__(self):
        if "POLYGON_KEY" not in os.environ:
            raise ValueError("Missing POLYGON_KEY key in env vars")

        self.cache = {}
        self.key = os.environ["POLYGON_KEY"]

    def _save_cache(self):
        if not os.path.exists("price_cache"):
            os.mkdir("price_cache")

        with open("price_cache/cache.pkl", "wb") as f:
            pickle.dump(self.cache, f)

    @staticmethod
    def ts_to_datetime(ts) -> str:
        return datetime.fromtimestamp(ts / 1000.0).strftime("%Y-%m-%d %H:%M")

    def _prefetch_tickers(self, symbol):
        closes = []

        print()
        with RESTClient(self.key) as client:
            cursor = arrow.get("2019-01-01")
            to = arrow.get("2020-12-01")

            while cursor < to:
                start, end = cursor.format("YYYY-MM-DD"), to.format("YYYY-MM-DD")
                resp = client.stocks_equities_aggregates(
                    symbol, 1, "minute", start, end, unadjusted=False
                )

                for result in resp.results:
                    closes.append((result["t"], result["c"]))

                print(
                    f"\rfetching {symbol}: {self.ts_to_datetime(result['t'])}", end=""
                )
                cursor = arrow.get(result["t"])

        print()
        return sorted(closes)


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import numpy as np

    quotes = QuotesFetcher()

    # load all news
    data_dir = "../pickle_news"
    df = pd.concat(
        [pd.read_pickle(f"{data_dir}/{filename}") for filename in os.listdir(data_dir)]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    symbols = set(df["symbol"])
    for symbol in tqdm(symbols, total=len(symbols)):
        times = df[df["symbol"] == symbol]["timestamp"].astype(int) / 10 ** 6

        closes = quotes._prefetch_tickers(symbol)
        for time in tqdm(times, total=len(times)):
            for close in closes:

                if close[0] > time:
                    quotes.cache[f"{symbol}{time}"] = close[1]
                    break

    quotes._save_cache()
