import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gather_env import GatheringEnv
from PGagent import PGagent, social_agent, newPG, IAC, social_IAC, Centralised_AC
from network import socialMask, Centralised_Critic
from copy import deepcopy
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = GatheringEnv(2)  # gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)

agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device}
# agentParam =

model_name = "pg_social"
file_name = "/Users/xue/Desktop/Social Law/saved_weight/" + model_name
save_eps = 10
ifsave_model = True
logger = Logger('./logs4')

class Agents():
    def __init__(self,agents,state_dim):
        self.num_agent = len(agents)
        self.agents = agents
        self.critic = Centralised_Critic(state_dim)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(),lr=0.01)
        self.lr_schedulerC = torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=1000, gamma=0.9, last_epoch=-1)
        for i in self.agents:
            i.critic = self.critic

    def choose_action(self,state):
        actions = []
        for agent, s in zip(self.agents, state):
            actions.append(agent.choose_action(s).detach().numpy())
        return actions

    def td_err(self, s, r, s_):
        s = torch.Tensor(s).reshape((1,-1)).unsqueeze(0)
        s_ = torch.Tensor(s_).reshape((1,-1)).unsqueeze(0)
        v = self.critic(s)
        v_ = self.critic(s_).detach()
        return r + 0.9*v_ - v

    def LearnCenCritic(self, s, r, s_):
        td_err = self.td_err(s,r,s_)
        # m = torch.log(self.agents.act_prob[a[0]]*self.agents.act_prob[a[1]])
        loss = torch.square(td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_schedulerC.step()

    def update(self, state, reward, state_, action):
        td_err = self.td_err(state,sum(reward),state_)
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a,td_err)
        self.LearnCenCritic(state,sum(reward),state_)

class Social_Agents():
    def __init__(self,agents,agentParam):
        self.Law = social_agent(agentParam)
        self.agents = agents
        self.n_agents = len(agents)

    def select_masked_actions(self, state):
        actions = []
        for i, ag in zip(range(self.n_agents), self.agents):
            masks, prob_mask = self.Law.select_action(state[i])
            self.Law.prob_social.append(prob_mask)  # prob_social is the list of masks for each agent
            pron_mask_copy = prob_mask  # deepcopy(prob_mask)
            action, prob_indi = ag.select_masked_action(state[i], pron_mask_copy)
            self.Law.pi_step.append(prob_indi)  # pi_step is the list of unmasked policy(prob ditribution) for each agent
            actions.append(action)
        return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a)

    def update_law(self):
        self.Law.update(self.n_agents)

    def push_reward(self, reward):
        for i, ag in zip(range(self.n_agents), self.agents):
            ag.rewards.append(reward[i])
        self.Law.rewards.append(sum(reward))


def main():
    # agent = PGagent(agentParam)
    # writers = [writer = SummaryWriter('runs/fashion_mnist_experiment_1')]
    n_agents = 2
    # multiPG = independentAgent([PGagent(agentParam) for i in range(n_agents)])
    multiPG = Agents([Centralised_AC(8,400) for i in range(n_agents)],400)  # create PGagents as well as a social agent
    # multiPG = Social_Agents([social_IAC(8,400,agentParam) for i in range(n_agents)],agentParam)
    for i_episode in range(101):
        n_state, ep_reward = env.reset(), 0  # reset the env
        for t in range(1, 500):
            actions = multiPG.choose_action(n_state)  # agent.select_action(state)   #select masked actions for every agent
            # actions = multiPG.select_masked_actions(n_state)
            n_state_, n_reward, _, _ = env.step(actions)  # interact with the env
            if args.render:  # render or not
                env.render()
            # multiPG.push_reward(n_reward)  # each agent receive their own reward, the law receive the summed reward
            ep_reward += sum(n_reward)  # record the total reward
            multiPG.update(n_state, n_reward, n_state_, actions)
            # multiPG.update_law()
            n_state = n_state_

        running_reward = ep_reward
        # loss = multiPG.update_agents()  # update the policy for each PGagent
        # multiPG.update_law()  # update the policy of law
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            logger.scalar_summary("ep_reward", ep_reward, i_episode)

        # if i_episode % save_eps == 0 and i_episode > 1 and ifsave_model:
            # multiPG.save(file_name)
        #


if __name__ == '__main__':
    main()
