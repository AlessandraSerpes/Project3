#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
try:
    import wandb
except:
    pass
from agent import Agent
from dqn_model import DQN

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN, self).__init__(env)
        self.env = env
        self.args = args

        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.02

        self.gamma = 0.99
        self.learning_rate = 0.000025
        self.batch_size = 32
        self.memory = deque(maxlen=50000)
        self.num_episodes = 100000
        self.model_save_freq = 1000
        self.target_update_freq = 75
        self.rewards_buffer = deque(maxlen=500)

        self.steps = 0

        self.epsilon_decay_ep = int(self.num_episodes*self.epsilon_decay)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net = DQN().to(self.device)
        self.target_q_net = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        if args.test_dqn:
            print('loading trained model')
            self.q_net.load_state_dict(torch.load(args.model_path))

    def init_game_setting(self):
        pass
    
    def make_action(self, observation, test=True):
        if test or random.random() > self.epsilon:
            observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
            observation = observation.permute(0, 3, 1, 2)
            with torch.no_grad():
                q_values = self.q_net(observation)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.env.action_space.n)
        return action
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay_buffer(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def train_agent(self):
        wandb.init(project="dqn_project", config={
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_episodes": self.num_episodes
        }, mode=self.args.wandb_mode)
        wandb.watch(self.q_net, log="all")

        for episode in tqdm(range(self.num_episodes), desc="Training Progress"):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, _, _ = self.env.step(action)
                self.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                self.train_model()
                self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min)*max(0, self.epsilon_decay_ep - episode) / self.epsilon_decay_ep

            self.rewards_buffer.append(total_reward)
            average_reward = np.mean(self.rewards_buffer)

            
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay

            wandb.log({"episode": episode, 
                       "total_reward": total_reward, 
                       "average_reward": average_reward, 
                       "epsilon": self.epsilon})

            if episode % self.model_save_freq == 0:
                torch.save(self.q_net.state_dict(), f"{wandb.run.dir}/dqn_model_{episode}_{average_reward}.pth")
                wandb.save(f"dqn_model_{episode}_{average_reward}.pth")
                print(f"Episode {episode}: Model saved!")
            
            if episode % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
                print(f"Episode {episode}: Target network updated!")

    def train_model(self):
        if len(self.memory) < 5000:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer()

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.q_net(next_states).argmax(dim=1)
        
        next_q_values = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.smooth_l1_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        

        wandb.log({"loss": loss.item()})
