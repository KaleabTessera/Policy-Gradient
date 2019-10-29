import gym
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class PolicyValueNetwork(nn.Module):
    def __init__(self, input_size=8, hidden_size=128,
                 num_hidden_layers=6, policy_output_size=4, value_output_size=1):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        for i in np.arange(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(
                hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        # Define simple policy head
        self.policy = nn.Sequential(
            # nn.Linear(hidden_size,
            #   hidden_size), nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.BatchNorm2d(hidden_size,hidden_size),
            nn.Linear(hidden_size,
                      policy_output_size),
            # nn.Tanh()
            )

        # Define simple value head
        self.value = nn.Sequential(
            # nn.Linear(hidden_size,
            #   hidden_size), nn.ReLU(),
            nn.Linear(hidden_size,
                      value_output_size))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        for layer in self.hidden_layers:
            out = layer(out)
        # probs = Categorical(logits=self.policy(out))
        # probs = F.softmax(self.policy(out))
        # probs = F.softmax(self.policy(out),dim=-1)
        # out = nn.BatchNorm2d(out)
        probs = self.policy(out)
        # print(f"probs:{probs}")
        return probs, self.value(out)

    def act(self, state):
        # state = torch.from_numpy(state).float().to(device)
        action_distribution, value = self.forward(state)
        # action = action_distribution.sample()
        # return action.item(), action_distribution.log_prob(action), value,action_distribution
        return action_distribution, value


def compute_returns(rewards, gamma):
    r = 0
    returns = []
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r
        returns.insert(0, r)
    returns = np.array(returns)
    mean = returns.mean(axis=0)
    std = returns.std(axis=0)
    returns = (returns - mean)/std
    return returns


def reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                               number_episodes,
                               gamma, verbose=False):

    max_episode_length = 1000
    scores = []
    policy_model = policy_model.to(device)
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    scores_deque = deque(maxlen=100)
    for e in np.arange(start=1, stop=number_episodes+1):
        saved_probs = []
        saved_value = []
        rewards = []
        policy_loss = []
        state = env.reset()
        for i in np.arange(start=1, stop=max_episode_length+1):
            # action, action_log_probs, value,action_distribution = policy_model.act(state)
            # print(state)
            # print(type(state))
            state = torch.from_numpy(state).float().to(device)
            action_log_probs, value = policy_model.act(state)
            action_log_probs = action_log_probs.detach().cpu()
            # print(action_log_probs)
            # action = np.random.choice(4, p=action_log_probs)
            # print(action)
            state, reward, done, _ = env.step(action_log_probs)
            rewards.append(reward)
            saved_value.append(value)
            # saved_probs.append(action_log_probs.mean().unsqueeze(0))
            saved_probs.append(action_log_probs.mean().unsqueeze(0))
            if done:
                break

        total_rewards = sum(rewards)
        scores.append(total_rewards)
        scores_deque.append(total_rewards)

        G = compute_returns(rewards, gamma)
        G = torch.from_numpy(G).float().to(device)

        total_value = torch.cat(saved_value)
        # print(saved_probs)
        saved_probs = torch.cat(saved_probs).to(device)
        # saved_probs = torch.cat(saved_probs)
        delta = G - total_value

        # Loss for policy
        # print(saved_probs)
        # print(delta)
        policy_loss = -torch.sum(saved_probs*delta.detach())
        # print(policy_loss)

        # Loss for value
        value_loss = 0.5*torch.sum(delta**2)

        # compute the composite loss
        loss = policy_loss + value_loss
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 50 == 0 and verbose:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                e, np.mean(scores_deque)))
    return policy_model, scores


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')

    print('Action space: ', env.action_space)
    print('Observation space: ', env.observation_space)

    # hyper-parameters
    gamma = 0.99
    learning_rate = 0.02
    seed = 401
    number_episodes = 1250

    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)

    policy_model = PolicyValueNetwork(input_size=24)
    policy_model = policy_model.to(device)

    net, scores = reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                                             number_episodes,
                                             gamma, verbose=True)

    # policy_model = PolicyValueNetwork()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(policy_model.state_dict(),
               f"./models/model_bipedal_{timestr}.pth")
    env = gym.wrappers.Monitor(
        env, './video_bipedal/', force=True)
    state = env.reset()
    for t in range(2000):
        state = torch.from_numpy(state).float().to(device)
        # dist, value = net(state)
        # print(state)
        # print(type(state))
        action_log_probs, value = policy_model.act(state)
        action_log_probs = action_log_probs.detach().cpu()
        # action = env.action_space.sample()
        # print(action)
        env.render()
        state, reward, done, _ = env.step(action_log_probs)
        if done:
            state = env.reset()
    env.close()
