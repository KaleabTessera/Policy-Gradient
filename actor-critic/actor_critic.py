from collections import namedtuple, deque
import copy
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
from ddpg_agent import Agent
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

# Code was based on https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal
# With modifications made to the actor and the critic.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ddpg(n_episodes=2000, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
        # while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        else:
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True


        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
            i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG')
    parser.add_argument(
        "--load-model", action="store_true", default=False, help="Load a trained model or start from scratch"
    )
    print(f"Device :{device}")

    env = gym.make('BipedalWalker-v2')

    print('Action space: ', env.action_space)
    print('Observation space: ', env.observation_space)

    seed = 10

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = parser.parse_args()
    # hyper-parameters
    number_episodes = 2500
    max_iter = 700
    env = gym.make('BipedalWalker-v2')
    env = gym.wrappers.Monitor(
        env, './bipedal_video/',  video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

    agent = Agent(
        state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=seed)
    if(args.load_model is True):
        print("Loading")
        agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    
    scores = ddpg(number_episodes, max_iter)
    fig = plt.figure()
    plt.title("Score (Returns) per episode")
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('Score_DDPG.png')
    # plt.show()
       

   
    state = env.reset()
    agent.reset()
    while True:
        action = agent.act(state)
        env.render()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

    env.close()
