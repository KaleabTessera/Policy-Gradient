import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn

import random
from collections import deque
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.softmax(x)

    def act(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        log_probs = self.forward(state)
        action_distribution = torch.distributions.Categorical(log_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    return sum(r*(gamma**i) for i, r in enumerate(rewards))


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    scores = []
    policy_model = policy_model.to(device)
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    scores_deque = deque(maxlen=100)
    for e in np.arange(start=1, stop=number_episodes+1):
        saved_probs = []
        rewards = []
        policy_loss = []
        state = env.reset()
        for i in np.arange(start=1, stop=max_episode_length+1):
            action, action_log_probs = policy_model.act(state)
            saved_probs.append(action_log_probs)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        total_rewards = sum(rewards)
        scores.append(total_rewards)
        scores_deque.append(total_rewards)
        G = compute_returns(rewards, gamma)
        # saved_probs = torch.cat(saved_probs)
        # policy_loss = -torch.sum(saved_probs*G)
        policy_loss.extend([-log_prob*G for log_prob in saved_probs])
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # report the score to check that we're making progress
        if e % 50 == 0 and verbose:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                e, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 495.0 and verbose:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                e, np.mean(scores_deque)))
            break
    return policy_model, scores

# This function adapted from https://github.com/andrecianflone/rl_at_ammi


def compute_returns_naive_baseline(rewards, gamma):
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


def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)

    scores = []
    policy_model = policy_model.to(device)
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    scores_deque = deque(maxlen=100)
    for e in np.arange(start=1, stop=number_episodes+1):
        saved_probs = []
        rewards = []
        policy_loss = []
        state = env.reset()
        for i in np.arange(start=1, stop=max_episode_length+1):
            action, action_log_probs = policy_model.act(state)
            saved_probs.append(action_log_probs)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        total_rewards = sum(rewards)
        scores.append(total_rewards)
        scores_deque.append(total_rewards)
        G = compute_returns_naive_baseline(rewards, gamma)
        G = torch.from_numpy(G).float().to(device)
        saved_probs = torch.cat(saved_probs)
        policy_loss = -torch.sum(saved_probs*G)
        # policy_loss.extend([-log_prob*G for log_prob in saved_probs])
        # policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
    # # report the score to check that we're making progress
    if e % 50 == 0 and verbose:
        print('Episode {}\tAverage Score: {:.2f}'.format(
            e, np.mean(scores_deque)))
    return policy_model, scores


def run_reinforce():
    env = gym.make('CartPole-v1')
    policy_model = SimplePolicy(
        s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
    policy, scores = reinforce(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
    # Plot learning curve
    window_size = 50
    moving_avg = moving_average(scores, window_size)
    plt.plot(scores, label='Score')
    plt.plot(
        moving_avg, label=f'Moving Average (w={window_size})', linestyle='--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('REINFORCE learning curve - CartPole-v1')
    plt.legend()
    plt.savefig('q1_reinforce_learning_curve.png')
    plt.show()


def investigate_variance_in_reinforce():
    env = gym.make('CartPole-v1')
    seeds = np.random.randint(1000, size=5)
    policy_model = SimplePolicy(
        s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)

    all_scores_over_runs = []
    window_size = 50
    for seed in seeds:
        _, score = reinforce(env=env, policy_model=policy_model, seed=int(seed), learning_rate=1e-2,
                             number_episodes=1500,
                             max_episode_length=1000,
                             gamma=1.0,
                             verbose=False)
        all_scores_over_runs.append(score)
        print(f"Reinforce - Seed:  {seed}  completed.")

    moving_avg_over_runs = np.array(
        [moving_average(score, 50) for score in all_scores_over_runs])
    mean = moving_avg_over_runs.mean(axis=0)
    std = moving_avg_over_runs.std(axis=0)

    plt.plot(mean, '-', color='blue')
    x = np.arange(1, len(mean)+1)
    plt.fill_between(x, mean-std, mean+std, color='blue', alpha=0.2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('REINFORCE averaged over 5 seeds')
    plt.savefig('q1_reinforce_5_runs_average.png')
    plt.show()

    return mean, std


def run_reinforce_with_naive_baseline(mean, std):
    env = gym.make('CartPole-v1')

    np.random.seed(53)
    seeds = np.random.randint(1000, size=5)
    policy_model = SimplePolicy(
        s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)

    all_scores_over_runs = []
    window_size = 50
    for seed in seeds:
        _, score = reinforce_naive_baseline(env=env, policy_model=policy_model, seed=int(seed), learning_rate=1e-2,
                                            number_episodes=1500,
                                            max_episode_length=1000,
                                            gamma=1.0,
                                            verbose=False)
        all_scores_over_runs.append(score)
        print(f"Reinforce with Naive Baseline - Seed:  {seed}  completed.")

    moving_avg_over_runs = np.array(
        [moving_average(score, 50) for score in all_scores_over_runs])
    mean_baseline = moving_avg_over_runs.mean(axis=0)
    std_baseline = moving_avg_over_runs.std(axis=0)

    # Reinforce
    plt.plot(mean, '-', color='blue', label='REINFORCE')
    x = np.arange(1, len(mean)+1)
    plt.fill_between(x, mean-std, mean+std, color='blue', alpha=0.2)

    # Reinforce with learned baseline
    plt.plot(mean_baseline, '-', color='orange',
             label='REINFORCE with baseline')
    x = np.arange(1, len(mean)+1)
    plt.fill_between(x, mean_baseline-std_baseline,
                     mean_baseline+std_baseline, color='orange', alpha=0.2)

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('REINFORCE vs REINFORCE with Baseline (averaged over 5 seeds)')
    plt.legend()
    plt.savefig('q1_reinforce_vs_reinforce_with_baseline_5_runs_average.png')
    plt.show()


if __name__ == '__main__':
    run_reinforce()
    mean, std = investigate_variance_in_reinforce()
    run_reinforce_with_naive_baseline(mean, std)
