import pandas as pd
import arrow
import numpy as np
import datetime as dt
from collections import Counter
import requests
import os
from tqdm import tqdm
import joblib
from sklearn.preprocessing import MinMaxScaler

key = os.environ["POLYGON_KEY"]
url = "https://api.polygon.io/v2/aggs/ticker"
params = f"?unadjusted=false&sort=asc&limit=5000&apiKey={key}"

df = pd.read_pickle("../cache/hist_options.pkl")
print(df.head())
day = arrow.get(df["Time"].iloc[0].format("YYYY-MM-DD"))
counter = Counter()


def get_prices(date):
    next_day = date.shift(days=1).format("YYYY-MM-DD")
    today = date.format("YYYY-MM-DD")

    # aggregate from polygon
    r = requests.get(f"{url}/{row['Ticker']}/range/1/minute/{today}/{next_day}{params}")
    if r.status_code != 200:
        print(f"failed to fetch quotes for {row['Ticker']}")
        exit()

    # find price at bod and current
    start_of_day = arrow.get(f"{today} 09:30")
    s_price, now_price = None, None
    for quote in r.json()["results"]:
        if s_price is None and arrow.get(quote["t"]) > start_of_day:
            s_price = quote["c"]

        if now_price is None and arrow.get(quote["t"]) > row["Time"]:
            now_price = quote["c"]

    if s_price is None or now_price is None:
        print(f'price cannot be `None` for {row["Ticker"]} on {row["Time"]}')
        exit()

    return s_price, now_price


data = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # new day - reset counter
    if arrow.get(row["Time"].format("YYYY-MM-DD")) > day:
        counter = Counter()

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

    # how much that stock has moved that day
    s_price, now_price = get_prices(row["Time"])

    # encode C/P, how far OTM, and how far price moved that day
    if row["C/P"] == "Call":
        entry.append(row["Strike"] / row["Spot"] - 1)
        entry.append(0)
        entry.append(now_price / s_price - 1)
    else:
        entry.append(row["Spot"] / row["Strike"] - 1)
        entry.append(1)
        entry.append(s_price / now_price - 1)

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
    entry.append(row["Volume"] / row["OI"])

    # size relative to OI
    entry.append(row["Qty"] / row["OI"])

    data.append(entry)

# scale
data = np.array(data)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
joblib.dump(scaler, "scaler.gz")

np.save("data.npy", data)
print(data.shape)
