from model.dqn_agent import DQNAgent
import gym
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from utils.trader import Trader


class TraderEnv(object):
    def __init__(self):
        print("setting up trading env")
        df = pd.read_pickle("cache/encoded_rows.pkl")
        encoded = np.load("cache/unscaled_data.npy").astype(np.float32)

        self.trader = Trader()
        self.current_step = 1
        valid_tickers = self.trader.quotes.valid_tickers

        # filter valid tickers
        valid_rows, valid_x = [], []
        for idx, row in df.iterrows():
            if row["Ticker"] in valid_tickers:
                valid_rows.append(row)
                valid_x.append(encoded[idx])

        df = pd.DataFrame(valid_rows)
        encoded = np.array(valid_x)

        # only use subset of data
        split = int(0.4 * len(encoded))
        df, encoded = df.iloc[split:], encoded[split:]

        split = int(0.6 * len(encoded))
        encoded, encoded_test = encoded[:split], encoded[split:]
        self.df, self.df_test = df.iloc[:split], df.iloc[split:]

        # scale
        scaler = MinMaxScaler()
        scaler.fit(encoded)
        self.encoded, self.encoded_test = scaler.transform(encoded), scaler.transform(
            encoded_test
        )
        joblib.dump(scaler, "cache/dqn_scaler.gz")

    def step(self, action):
        ###
        # Check for new trading day
        ###

        row = self.df.iloc[self.current_step]
        if action == 0:
            current_price = row["Spot"]
            expiry = row["Expiry"].format("YYYY-MM-DD")
            ticker = row["Ticker"]
            self.trader.trade_on_signal(ticker, "BULLISH", current_price, expiry)

        next_state = self.encoded[self.current_step]
        self.current_step += 1
        reward = self.trader.current_reward
        done = reward < -50 and self.current_step == len(self.encoded)

        print(reward)

        return next_state, reward, done

    def reset(self):
        self.trader = Trader()
        self.current_step = 1
        return self.encoded[0]


seed = 777


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 100000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

env = TraderEnv()
agent = DQNAgent(
    env,
    env.encoded.shape[1],
    memory_size,
    batch_size,
    target_update,
    epsilon_decay,
)
agent.train(num_frames)
