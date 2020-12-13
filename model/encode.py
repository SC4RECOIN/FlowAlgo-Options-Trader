import pandas as pd
import pandas_ta as ta
import arrow
import numpy as np
import datetime as dt
from collections import Counter
import requests
import os
from tqdm import tqdm
import joblib
import json
from sklearn.preprocessing import MinMaxScaler

key = os.environ["POLYGON_KEY"]
url = "https://api.polygon.io/v2/aggs/ticker"
params = f"?unadjusted=true&sort=asc&limit=5000&apiKey={key}"

df = pd.read_pickle("../cache/hist_options.pkl")
print(df.head())


# prefetch aggregates from ploygon
def prefetch_agg(tickers):
    if os.path.exists("quotes.json"):
        with open("quotes.json") as f:
            return json.load(f)

    quotes = {}
    for ticker in tqdm(tickers, total=len(tickers), desc="Fetching quotes"):
        quotes[ticker] = {}

        try:
            r = requests.get(
                f"{url}/{ticker}/range/1/day/2017-04-01/2020-12-01{params}"
            )
            if r.status_code != 200:
                raise Exception(f"failed to fetch quotes for {ticker}")

            for quote in r.json()["results"]:
                day = arrow.get(quote["t"]).format("YYYY-MM-DD")
                quotes[ticker][day] = quote

        except Exception as e:
            print(e)

    with open("quotes.json", "w") as f:
        json.dump(quotes, f, indent=4)

    return quotes


tickers = list(set(df["Ticker"]))
quotes = prefetch_agg(tickers)
bad_tickers = []

# calc TA
for ticker in tickers:
    try:
        quotes_df = pd.DataFrame(quotes[ticker]).transpose()
        quotes_df["rsi10"] = ta.rsi(quotes_df["c"], length=10)
        quotes[ticker] = quotes_df.to_dict(orient="index")
    except:
        bad_tickers.append(ticker)

day = arrow.get(df["Time"].iloc[0].format("YYYY-MM-DD"))
counter = Counter()

data, dates, tickers = [], [], []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # new day - reset counter
    if arrow.get(row["Time"].format("YYYY-MM-DD")) > day:
        counter = Counter()

    if row["Ticker"] in bad_tickers:
        continue

    try:
        entry = []

        # keep track of ticker c/p and overall c/p ratio
        counter[f"{row['C/P']}{row['Ticker']}"] += 1
        counter[row["C/P"]] += 1

        # occurence of calls and puts and c/p ratio
        entry.append(counter[f"Call{row['Ticker']}"])
        entry.append(counter[f"Put{row['Ticker']}"])
        entry.append(counter["Call"] / max(counter["Put"], 1))

        # seconds since start of day
        bod = arrow.get(row["Time"].format("YYYY-MM-DD"))
        diff = (row["Time"] - bod).seconds
        entry.append(diff)

        # seconds to expiry
        entry.append((row["Expiry"] - row["Time"]).seconds)

        # encode C/P, how far OTM
        if row["C/P"] == "Call":
            entry.append(row["Strike"] / row["Spot"] - 1)
            entry.append(0)

            # how far the stock went up that day
            d = row["Time"].format("YYYY-MM-DD")
            chg = row["Spot"] / quotes[row["Ticker"]][d]["o"] - 1

            if abs(chg) > 0.3:
                raise Exception(
                    f"Irregular price movement for {row['Ticker']} on {d} ({chg})"
                )
            entry.append(chg)
        else:
            entry.append(row["Spot"] / row["Strike"] - 1)
            entry.append(1)

            # how far the stock went down that day
            d = row["Time"].format("YYYY-MM-DD")
            chg = quotes[row["Ticker"]][d]["o"] / row["Spot"] - 1

            if abs(chg) > 0.3:
                raise Exception(
                    f"Irregular price movement for {row['Ticker']} on {d} ({chg})"
                )
            entry.append(chg)

        yesterday = row["Time"].shift(days=-1)

        # Sunday
        if yesterday.weekday() == 6:
            yesterday = yesterday.shift(days=-2)

        rsi = quotes[row["Ticker"]][yesterday.format("YYYY-MM-DD")]["rsi10"]
        entry.append(rsi)

        # order type
        if row["Type"] == "SWEEP":
            entry.append(0)
        elif row["Type"] == "BLOCK":
            entry.append(1)
        else:
            entry.append(2)

        entry.append(int(row["Unusual"]))
        entry.append(row["Premium"])

        # very active if daily volume exceeds OI
        entry.append(row["Volume"] / max(1, row["OI"]))

        # size relative to OI
        entry.append(row["Qty"] / max(1, row["OI"]))

        data.append(entry)
        dates.append(row["Time"].timestamp)
        tickers.append(row["Ticker"])

    except Exception as e:
        print(e)

with open("data.json", "w") as f:
    json.dump({"tickers": tickers, "dates": dates}, f, indent=4)

# scale
data = np.array(data)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
joblib.dump(scaler, "scaler.gz")

np.save("data.npy", data)
print(data.shape)

assert len(data) == len(dates)
assert len(data) == len(tickers)
