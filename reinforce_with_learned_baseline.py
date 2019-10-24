import gym
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyValueNetwork(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
        return dist, value


def compute_returns(rewards, gamma):
    raise NotImplementedError
    return returns


def reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                               number_episodes,
                               gamma, verbose=False):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)

    # if episode % 50 == 0 and verbose:
    #     print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
    raise NotImplementedError
    return policy_value_net, scores


def main():
    env = gym.make('LunarLander-v2')
    print('Action space: ', env.action_space)
    print('Observation space: ', env.observation_space)

    # hyper-parameters
    gamma = 0.99
    learning_rate = 0.02
    # seed = 214
    seed = 401
    number_episodes = 1250
    policy_model = PolicyValueNetwork()

    net, scores = reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                                             number_episodes,
                                             gamma, verbose=True)

    state = env.reset()
    for t in range(2000):
        state = torch.from_numpy(state).float().to(device)
        dist, value = net(state)
        action = dist.sample().item()
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
    env.close()


if __name__ == '__main__':
    main()
