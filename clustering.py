import numpy as np
import pandas as pd
import arrow
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MeanShift, DBSCAN
import matplotlib.pyplot as plt
from joblib import dump, load
from utils.trader import Trader

random_state = 42


def visualize(encodings):
    print("dimensionality reduction and visualization...")
    x_2d = TSNE(n_components=2).fit_transform(encodings)
    np.save("cache/tsne.npy", x_2d)
    plt.scatter(x_2d[:, 0], x_2d[:, 1])
    plt.show()


def clustering(encodings, df, method, params):
    clusters = method(**params).fit(encodings)
    dump(clusters, 'clustering.joblib')

    # check each cluster for profitability
    n_clusters = len(set(clusters.labels_))

    # noise labeled as -1
    if -1 in clusters.labels_:
        n_clusters -= 1

    best = 0
    best_score = 0

    print(f"{n_clusters} number of clusters")
    for i in range(n_clusters):
        trader = Trader()
        day = arrow.get(df["Time"].iloc[0].format("YYYY-MM-DD"))

        for idx, row in df.iterrows():
            current_day = arrow.get(row["Time"].format("YYYY-MM-DD"))

            # new day, check expiries
            if current_day > day:
                trader.eod(day.format("YYYY-MM-DD"))
                day = current_day

            # if target cluster, buy
            if clusters.labels_[idx] == i:
                current_price = row["Spot"]
                expiry = row["Expiry"].format("YYYY-MM-DD")
                ticker = row["Ticker"]
                trader.trade_on_signal(ticker, "BULLISH", current_price, expiry)

        reward = trader.current_reward
        print(f"cluster {i}\treturn: {reward:.2f}%")

        if reward > best_score:
            best_score = reward
            best = i 
        
    print(f"best {best}")
    return best


def test(encodings, df, target_cluster):
    clusters = load('clustering.joblib')

    trader = Trader()
    day = arrow.get(df["Time"].iloc[0].format("YYYY-MM-DD"))
    print(f"start {day}")

    for idx, row in df.iterrows():
        current_day = arrow.get(row["Time"].format("YYYY-MM-DD"))

        # new day, check expiries
        if current_day > day:
            trader.eod(day.format("YYYY-MM-DD"))
            day = current_day

        # if target cluster, buy
        if clusters.labels_[idx] == target_cluster:
            current_price = row["Spot"]
            expiry = row["Expiry"].format("YYYY-MM-DD")
            ticker = row["Ticker"]
            trader.trade_on_signal(ticker, "BULLISH", current_price, expiry)

    print(f"end {current_day}")
    print(f"cluster {target_cluster}\treturn: {trader.current_reward:.2f}%")


def main(encodings, df):
    return clustering(encodings, df, KMeans, {"n_clusters":100})

if __name__ == "__main__":
    df = pd.read_pickle("cache/encoded_rows.pkl")
    print(df.head())

    encoded = np.load("cache/data.npy").astype(np.float32)
    assert len(encoded) == len(df)

    trader = Trader()
    valid_tickers = trader.quotes.valid_tickers

    # filter valid tickers
    valid_rows, valid_x = [], []
    for idx, row in df.iterrows():
        if row["Ticker"] in valid_tickers:
            valid_rows.append(row)
            valid_x.append(encoded[idx])

    print(encoded.shape)
    df = pd.DataFrame(valid_rows)
    encoded = np.array(valid_x)
    assert len(encoded) == len(df)

    split = int(0.6 * len(encoded))
    encoded, encoded_test = encoded[:split], encoded[split:]
    df, df_test = df.iloc[:split], df.iloc[split:]
    print(encoded.shape)

    target_cluster = main(encoded, df)
    test(encoded_test, df_test, target_cluster)
