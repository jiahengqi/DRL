import gym
import tensorflow as tf
import numpy as np
import os
from utils import *
from tqdm import trange
from model import dueling_Double_DQN


MAXSTEP=500
convergence_reward = 475
VERSION='nature'

if __name__ == "__main__":
    train_episodes = 5000  # 1000          # max number of episodes to learn from
    max_steps = MAXSTEP  # 200                # max steps in an episode
    gamma = 0.99  # future reward discount

    # agent parameters
    state_size = 4
    action_size = 2
    # training process
    rewards_list = []
    test_rewards_list = []
    show_every_steps = 100

    # Exploration parameters
    explore_start = 0.5  # exploration probability at start
    explore_stop = 0.01  # minimum S probability
    decay_rate = 0.0001  # expotentional decay rate for exploration prob

    # Network parameters
    hidden_size = 20  # number of units in each Q-network hidden layer


    # Memory parameters
    memory_size = 10000  # memory capacity
    batch_size = 32  # experience mini-batch size
    pretrain_length = batch_size  # number experiences to pretrain the memory



    # Initialize the simulation
    env = gym.make('CartPole-v1').env
    env.reset()
    # Take one random step to get the pole and cart moving
    state, reward, done, _ = env.step(env.action_space.sample())
    #TODO 指定网络参数和名字
    agent = dueling_Double_DQN(env,maxlen=10000,version=VERSION)
    #memory = Memory(max_size=memorSy_size)

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        # env.render()

        # Make a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            agent.store(state, action, reward, next_state, done)

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, done, _ = env.step(env.action_space.sample())
        else:
            # Add experience to memory
            agent.store(state, action, reward, next_state, done)
            state = next_state



    step = 0
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        #episode_total
        while t < max_steps:
            step += 1
            # Uncomment this next line to watch the training
            # env.render()
            action = agent.get_action(state)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps
                rewards_list.append((ep, total_reward))
                # Add experience to memory
                agent.store(state, action, reward, next_state, done)

                # Start new episode
                state=env.reset()
            else:
                # Add experience to memory
                agent.store(state, action, reward, next_state, done)
                state = next_state
                t += 1

            agent.update()

        test_rewards_list.extend(test_agent(agent, env, test_max_steps=MAXSTEP))
        cur_compute_len = min(100, len(test_rewards_list))
        mean_reward = np.mean(test_rewards_list[len(test_rewards_list) - cur_compute_len:])
        print('Episode: {}'.format(ep),
              'Mean test reward: {:.1f}'.format(mean_reward), )
        if mean_reward > convergence_reward:
            print(ep, "收敛")
            break



    reward_list = []
    test_max_steps = convergence_reward + 5

    state = env.reset()
    t = 0
    while True:
        env.render()
        action = agent.get_action(state,False)
        next_state, reward, done, _ = env.step(action)
        if done:
            break
        else:
            state = next_state
            t += 1
                
                
    print(t)