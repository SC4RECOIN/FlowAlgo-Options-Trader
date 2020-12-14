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
        try:
            with open("cache/prices.pkl", "rb") as f:
                self.cache = pickle.load(f)
        except FileNotFoundError:
            self.cache = {}
            print("missing price cache")

        self.key = os.environ["POLYGON_KEY"]

    def get_quote(self, symbol: str, date: str) -> float:
        return self.cache[symbol][date]

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, newvalue):
        self.cache[key] = newvalue

    def _save_cache(self):
        with open("../cache/eod_prices.json", "w") as f:
            json.dump(self.cache, f, indent=4)

        with open("../cache/eod_prices.pkl", "wb") as f:
            pickle.dump(self.cache, f)

    def fetch_quotes(self, symbol):
        closes = {}

        with RESTClient(self.key) as client:
            resp = client.stocks_equities_aggregates(
                symbol, 1, "day", "2017-01-01", "2020-12-01", unadjusted=True
            )

            for result in resp.results:
                day = arrow.get(result["t"]).format("YYYY-MM-DD")
                closes[day] = result["c"]

        self.quotes[symbol] = closes


if __name__ == "__main__":
    from tqdm import tqdm

    quotes = Quotes()
    with open("../cache/data.json") as f:
        data = json.load(f)

    # only take most frequent stocks
    symbols = Counter(data["tickers"]).most_common(400)
    symbols = [s[0] for s in symbols]

    # fetch minute data for each symbol
    for symbol in tqdm(symbols, total=len(symbols)):
        quotes.fetch_quotes(symbol)

    quotes._save_cache()
