import numpy as np
import pandas as pd
import torch
import os
import json
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
from tqdm import tqdm
import joblib
import pathlib
from sklearn.preprocessing import MinMaxScaler
from model.ppg import PPG, Memory
from utils.trader import Trader
import arrow

signals = ["buy", "sell"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(
    encodings,
    df,
    num_episodes=1000,
    actor_hidden_dim=32,
    critic_hidden_dim=256,
    minibatch_size=64,
    lr=0.0005,
    betas=(0.9, 0.999),
    lam=0.95,
    gamma=0.99,
    eps_clip=0.2,
    value_clip=0.4,
    beta_s=0.01,
    update_timesteps=10000,
    num_policy_updates_per_aux=32,
    epochs=1,
    epochs_aux=6,
    seed=None,
    save_every=5,
    load=False,
):
    state_dim = encodings.shape[1]
    num_actions = len(signals)

    memories = deque([])
    aux_memories = deque([])

    agent = PPG(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
    )

    if load:
        agent.load()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0
    reward = 0
    writer = SummaryWriter()

    for eps in tqdm(range(num_episodes), desc="episodes"):
        trader = Trader()
        day = arrow.get(df["Time"].iloc[0].format("YYYY-MM-DD"))
        done = False

        pbar = tqdm(range(len(encodings) - 1), total=len(encodings))
        for idx in pbar:
            # get row for price and date
            row = df.iloc[idx]
            current_day = arrow.get(row["Time"].format("YYYY-MM-DD"))
            time += 1

            # new day, check expiries
            if current_day != day:
                trader.eod(day.format("YYYY-MM-DD"))
                day = current_day

            state = encodings[idx]
            state = torch.from_numpy(state).to(device)
            action_probs, _ = agent.actor(state)
            value = agent.critic(state)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            if action == 0:
                current_price = row["Spot"]
                expiry = row["Expiry"].format("YYYY-MM-DD")
                ticker = row["Ticker"]
                trader.trade_on_signal(ticker, "BULLISH", current_price, expiry)

            reward = trader.current_reward

            if idx % 1000 == 0:
                pbar.set_description(f"current return {reward:.2f}%")

            if reward < -60:
                done = True

            next_state = encodings[idx + 1]
            memory = Memory(state, action, action_log_prob, reward, done, value)

            memories.append(memory)
            state = next_state

            if time % update_timesteps == 0:
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

            if done:
                break

        if eps % save_every == 0:
            agent.save()

        writer.add_scalar("reward", reward, eps)

    writer.flush()
    writer.close()


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

    # only use subset of data
    split = int(0.4 * len(encoded))
    df, encoded = df.iloc[split:], encoded[split:]

    split = int(0.6 * len(encoded))
    encoded, encoded_test = encoded[:split], encoded[split:]
    df, df_test = df.iloc[:split], df.iloc[split:]
    print(encoded.shape)

    # scale
    scaler = MinMaxScaler()
    scaler.fit(encoded)
    encoded, encoded_test = scaler.transform(encoded), scaler.transform(encoded_test)
    joblib.dump(scaler, "cache/cluster_scaler.gz")

    target_cluster = main(encoded, df)

    # train dqn model
    main(encoded, df, lr=0.001, critic_hidden_dim=128, gamma=0.999)