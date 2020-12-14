import numpy as np
import pandas as pd
import torch
import os
import json
from model.ppg import PPG, Memory, device
from torch.distributions import Categorical
from collections import deque, namedtuple
from tqdm import tqdm
from utils.trader import Trader
import datetime
import arrow

signals = ["BULLISH", "NEUTRAL", "BEARISH"]


def main(
    encodings,
    tickers, 
    dates,
    num_episodes=50000,
    max_timesteps=500,
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
    update_timesteps=5000,
    num_policy_updates_per_aux=32,
    epochs=1,
    epochs_aux=6,
    seed=None,
    render=False,
    render_every_eps=250,
    save_every=1000,
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

    for eps in tqdm(range(num_episodes), desc="episodes"):
        trader = Trader()

        for idx in range(len(encodings) - 1):
            time += 1

            state = encodings[idx]
            state = torch.from_numpy(state).to(device)
            action_probs, _ = agent.actor(state)
            value = agent.critic(state)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            trader.trade_on_signal(
               tickers, signals[action], dates[idx]
            )
            reward = trader.reward(dates[idx])

            next_state = encodings[idx + 1]
            memory = Memory(state, action, action_log_prob, reward, False, value)
            memories.append(memory)

            state = next_state

            if time % update_timesteps == 0:
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

        print(f"reward after episode: {reward:.2f}%")

        if eps % save_every == 0:
            agent.save()


if __name__ == "__main__":
    with open("model/data.json") as f:
        data = json.load(f)

    encoded = np.load("model/data.npy")
    assert len(encoded) == len(data['tickers'])
    assert len(encoded) == len(data['dates'])

    main(encoded, len(data['tickers'], len(data['dates'])
