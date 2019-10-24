import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn

import random
from collections import deque
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(s_size,h_size)
        self.fc2 = nn.Linear(h_size,a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    return sum(r*(gamma**i)  for i, r in enumerate(rewards))


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.seed(seed)
    episode = []
    for i in np.arange(start=1, stop=number_episodes+1):
        state = env.reset()
        for i in np.arange(start=1, stop=max_episode_length+1):
            action_probs = policy_model(torch.from_numpy(state).float().to(device))
            values, indices = action_probs.max(0)
            action = indices.item()
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break 
        
        for e in episode:
            state, action, reward = e
            first_occurrence = next(
                i for i, x in enumerate(episode) if (x[0] == state).all())
            
            rewards = (x[2] for x in episode[first_occurrence:])
            G = compute_returns(rewards,gamma)

            #Update weights

    # report the score to check that we're making progress
    # if episode % 50 == 0 and verbose:
    #     print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

    # if np.mean(scores_deque) >= 495.0 and verbose:
    #     print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
    #     break
    raise NotImplementedError
    return policy, scores


def compute_returns_naive_baseline(rewards, gamma):
    raise NotImplementedError
    return returns


def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)

    # # report the score to check that we're making progress
    # if episode % 50 == 0 and verbose:
    #     print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
    raise NotImplementedError
    return policy, scores


def run_reinforce():
    env = gym.make('CartPole-v1')
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
    policy, scores = reinforce(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
    # Plot learning curve


def investigate_variance_in_reinforce():
    env = gym.make('CartPole-v1')
    seeds = np.random.randint(1000, size=5)

    raise NotImplementedError

    return mean, std


def run_reinforce_with_naive_baseline(mean, std):
    env = gym.make('CartPole-v1')

    np.random.seed(53)
    seeds = np.random.randint(1000, size=5)
    raise NotImplementedError


if __name__ == '__main__':
    run_reinforce()
    mean, std = investigate_variance_in_reinforce()
    run_reinforce_with_naive_baseline(mean, std)
