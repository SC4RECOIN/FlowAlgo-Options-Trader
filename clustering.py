import numpy as np
import pandas as pd
import arrow
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from joblib import dump, load
from utils.trader import Trader
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

random_state = 42


def visualize(encodings):
    print("dimensionality reduction and visualization...")
    x_2d = TSNE(n_components=2).fit_transform(encodings)
    np.save("cache/tsne.npy", x_2d)
    plt.scatter(x_2d[:, 0], x_2d[:, 1])
    plt.show()


def clustering(encodings, df, method, params, topn=1):
    clusters = method(**params).fit(encodings)
    dump(clusters, "clustering.joblib")

    # check each cluster for profitability
    n_clusters = len(set(clusters.labels_))

    # noise labeled as -1
    if -1 in clusters.labels_:
        n_clusters -= 1

    scores = []

    for i in tqdm(range(n_clusters), total=n_clusters):
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

        scores.append(trader.current_reward)

    top = sorted(scores, reverse=True)[:topn]
    return [scores.index(s) for s in top]


def test(encodings, df, target_clusters):
    clusters = load("clustering.joblib")

    trader = Trader()
    day = arrow.get(df["Time"].iloc[0].format("YYYY-MM-DD"))
    print(f"start {day.format('YYYY-MM-DD')}")

    for idx, row in df.iterrows():
        current_day = arrow.get(row["Time"].format("YYYY-MM-DD"))

        # new day, check expiries
        if current_day > day:
            trader.eod(day.format("YYYY-MM-DD"))
            day = current_day

        # if target cluster, buy
        if clusters.labels_[idx] in target_clusters:
            current_price = row["Spot"]
            expiry = row["Expiry"].format("YYYY-MM-DD")
            ticker = row["Ticker"]
            trader.trade_on_signal(ticker, "BULLISH", current_price, expiry)

    print(f"end {current_day.format('YYYY-MM-DD')}")
    print(f"cluster {target_clusters}\treturn: {trader.current_reward:.2f}%")


def main(encodings, df, topn):
    return clustering(encodings, df, KMeans, {"n_clusters": 100}, topn)
    # return clustering(encodings, df, DBSCAN, {"eps": 0.3}, topn)
    # return clustering(encodings, df, AgglomerativeClustering, {"n_clusters": 75}, 1)


if __name__ == "__main__":
    df = pd.read_pickle("cache/encoded_rows.pkl")
    print(df.head())

    encoded = np.load("cache/unscaled_data.npy").astype(np.float32)
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

    step = 35000
    for i in range(70000, len(encoded), step):
        _encoded = encoded[i - 70000 : i]
        _df = df.iloc[i - 70000 : i]

        split = int(0.7 * len(_encoded))
        _encoded, _encoded_test = _encoded[:split], _encoded[split:]
        _df, _df_test = _df.iloc[:split], _df.iloc[split:]

        # scale
        scaler = MinMaxScaler()
        scaler.fit(_encoded)
        _encoded = scaler.transform(_encoded)
        _encoded_test = scaler.transform(_encoded_test)
        joblib.dump(scaler, "cache/cluster_scaler.gz")

        target_clusters = main(_encoded, _df, 1)
        test(_encoded_test, _df_test, target_clusters)
