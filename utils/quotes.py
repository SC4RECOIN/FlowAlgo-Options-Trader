from polygon import RESTClient
from datetime import datetime
import pickle
import arrow
import os
from collections import Counter
import json
import signal


class Quotes(object):
    def __init__(self):
        if not os.path.exists("price_cache/cache.pkl"):
            raise ValueError("Missing price cache")

        with open("price_cache/cache.pkl", "rb") as f:
            self.cache = pickle.load(f)

        self.key = os.environ["POLYGON_KEY"]

    def get_quote(self, symbol, timestamp):
        try:
            return self.cache[f"{symbol}{timestamp}"]
        except KeyError:
            start = arrow.get(timestamp)
            end = start.shift(days=1)
            timestamp *= 1000

            with RESTClient(self.key) as client:
                resp = client.stocks_equities_aggregates(
                    symbol,
                    1,
                    "minute",
                    start.format("YYYY-MM-DD"),
                    end.format("YYYY-MM-DD"),
                    unadjusted=False,
                )

                for result in resp.results:
                    if result["t"] > timestamp:
                        return result["c"]

        raise Exception(f"No price found for {symbol}")

    def __getitem__(self, key):
        return self.cache[key]


class QuotesFetcher(object):
    def __init__(self):
        if "POLYGON_KEY" not in os.environ:
            raise ValueError("Missing POLYGON_KEY key in env vars")

        self.cache = {}
        self.key = os.environ["POLYGON_KEY"]

    def _save_cache(self):
        if not os.path.exists("../price_cache"):
            os.mkdir("../price_cache")

        with open("../price_cache/cache.pkl", "wb") as f:
            pickle.dump(self.cache, f)

    @staticmethod
    def ts_to_datetime(ts) -> str:
        return datetime.fromtimestamp(ts / 1000.0).strftime("%Y-%m-%d %H:%M")

    def _prefetch_tickers(self, symbol):
        closes = {}

        print()
        with RESTClient(self.key) as client:
            cursor = arrow.get("2017-04-01")
            to = arrow.get("2020-12-01")

            while cursor < to:
                start, end = cursor.format("YYYY-MM-DD"), to.format("YYYY-MM-DD")
                try:
                    resp = client.stocks_equities_aggregates(
                        symbol, 1, "minute", start, end, unadjusted=False
                    )

                    for result in resp.results:
                        closes[result["t"]] = result["c"]

                    t = self.ts_to_datetime(result["t"])
                    print(f"\rfetching {symbol}: {t}", end="")
                    cursor = arrow.get(result["t"])
                except:
                    return closes

        print()
        return closes


if __name__ == "__main__":
    from tqdm import tqdm

    quotes = QuotesFetcher()
    with open("../model/data.json") as f:
        data = json.load(f)

    # only take 200 most frequent stocks
    symbols = Counter(data["tickers"]).most_common(200)
    symbols = [s[0] for s in symbols]

    # round times to nearest minute for lookup (convert back to ms)
    times = [(int(t / 60) + 1) * 60000 for t in set(data["dates"])]
    unform_times = set(data["dates"])

    # fetch minute data for each symbol
    for symbol in tqdm(symbols, total=len(symbols)):

        try:
            closes = quotes._prefetch_tickers(symbol)

            for time, _t in tqdm(zip(times, unform_times), total=len(times)):
                if time in closes:
                    quotes.cache[f"{symbol}{_t}"] = closes[time]

        except Exception as e:
            print(f"failed for {symbol}: {e}")

    quotes._save_cache()
