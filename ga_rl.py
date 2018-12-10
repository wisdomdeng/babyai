#!/usr/bin/env python3

import copy
import math
from functools import reduce
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

import gym
import babyai
from babyai.utils.demos import load_demos, transform_demos
from gym_minigrid.minigrid import MiniGridEnv

def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def print_model_info(model):
    modelSize = 0
    for p in model.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print(str(model))
    print('Total model size: %d' % modelSize)

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    arr = torch.from_numpy(arr).float()
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

class Model(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(147, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )

        self.rnn = nn.GRUCell(input_size=128, hidden_size=128)

        # GRU embedding to action
        self.action_probs = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.LeakyReLU(),
            nn.LogSoftmax(dim=1)
        )

        self.apply(init_weights)

    def predict_action(self, img, memory):
        batch_size = img.size(0)

        x = img.view(batch_size, -1)
        x = self.encoder(x)

        memory = self.rnn(x, memory)
        action_probs = self.action_probs(memory)
        dist = Categorical(logits=action_probs)

        return dist, memory

##############################################################################

env = gym.make('BabyAI-GoToRedBall-v0')

num_actions = env.action_space.n
print('num actions:', num_actions)

max_steps = env.max_steps
print('max episode steps:', max_steps)

max_episodes = 8192
num_episodes = 0
eps_idx = 0

# Done indicates that we become done after the current step
obss = np.zeros(shape=(max_episodes, max_steps, 147))
actions = np.zeros(shape=(max_episodes, max_steps, 1), dtype=np.long)
rewards = np.zeros(shape=(max_episodes, max_steps, 1))
active = np.zeros(shape=(max_episodes, max_steps, 1))

# Environment to collect experience from
env = gym.make('BabyAI-GoToRedBall-v0')





def evaluate(model, seed=0, num_episodes=100):
    env = gym.make('BabyAI-GoToRedBall-v0')

    num_success = 0

    env.seed(seed)

    for i in range(num_episodes):
        obs = env.reset()

        memory = Variable(torch.zeros([1, 128])).cuda()

        while True:

            obs = make_var(obs['image']).unsqueeze(0)

            dist, memory = model.predict_action(obs, memory)
            action = dist.sample()

            obs, reward, done, info = env.step(action)

            if done:
                if reward > 0:
                    num_success += 1
                break

    return num_success / num_episodes




def gen_experience(env, model, num_rollouts=8):
    global eps_idx
    global num_episodes

    for i in range(num_rollouts):
        #print(eps_idx)

        # Reset active flags for episode
        active[eps_idx] = 0

        obs = env.reset()

        memory = Variable(torch.zeros([1, 128])).cuda()

        for step_idx in range(max_steps):
            active[eps_idx, step_idx] = 1

            obs = obs['image']
            obss[eps_idx, step_idx] = obs.reshape((147,))

            obs = make_var(obs).unsqueeze(0)
            dist, memory = model.predict_action(obs, memory)
            action = dist.sample()

            actions[eps_idx, step_idx] = action

            obs, reward, done, info = env.step(action)

            if done:
                break

        for i in reversed(range(step_idx+1)):
            rewards[eps_idx, i] = reward
            reward *= 0.99

        num_episodes = max(num_episodes, eps_idx+1)
        eps_idx = (eps_idx + 1) % max_episodes




def reinforce(optimizer, model, batch_size=256):
    batch_size = min(batch_size, num_episodes)

    # Select a valid episode index in function of the batch size
    eps_idx = np.random.randint(0, num_episodes - batch_size + 1)

    # Get the observations, actions and done flags for this batch
    obs_batch = obss[eps_idx:(eps_idx+batch_size)]
    act_batch = actions[eps_idx:(eps_idx+batch_size)]
    rew_batch = rewards[eps_idx:(eps_idx+batch_size)]
    active_batch = active[eps_idx:(eps_idx+batch_size)]

    obs_batch = make_var(obs_batch)
    act_batch = make_var(act_batch)
    rew_batch = make_var(rew_batch)
    active_batch = make_var(active_batch)

    memory = Variable(torch.zeros([batch_size, 128])).cuda()

    policy_loss = 0

    mean_rew = rew_batch.mean()

    # For each step
    # We will iterate until the max demo len (or until all demos are done)
    for step_idx in range(max_steps):
        active_step = active_batch[:, step_idx, :]

        if active_step.sum().item() == 0:
            break

        obs_step = obs_batch[:, step_idx, :]
        act_step = act_batch[:, step_idx, :]
        rew_step = rew_batch[:, step_idx, :]

        dist, memory = model.predict_action(obs_step, memory)

        #print(act_step.size())

        log_prob = dist.log_prob(act_step.squeeze(1))

        #print(log_prob.size())
        #print(rew_step.size())
        #print(active_step.size())

        #rew_step = rew_step - mean_rew
        policy_loss += (-log_prob * rew_step * active_step).sum()

    #print(policy_loss)

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()











model = Model(num_actions)
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-4)

best_score = 0
best_model = model





for i in range(10000):
    #model = copy.deepcopy(best_model)

    print('  gen experience')
    gen_experience(env, model)

    print('  reinforce')
    reinforce(optimizer, model, batch_size=16)


    if i % 20 == 0:
        print('evaluate')
        s = evaluate(model)
        print('#{}: {:.3f}'.format(i+1, s))

    #if s > best_score:
    #    print('new best score: {:.3f}'.format(s))
    #    best_model = copy.deepcopy(model)
    #    best_score = s




"""
for i in range(100):
    model = Model(num_actions)
    model.cuda()
    s = evaluate(model)

    print('#{}: {:.3f}'.format(i+1, s))

"""

# TODO; start with 10 random models, evaluate them
# perform reinforce based on

# TODO: use SGD optimizer

# TODO: gather off-policy experience
