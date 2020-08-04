import argparse
import gym
import numpy as np
from itertools import count
import random
import torch
# import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from gather_env import GatheringEnv
from cleanup import CleanupEnv
from harvest import HarvestEnv
from PGagent import PGagent, social_agent, newPG, IAC, social_IAC
import matplotlib.pyplot as plt
from network import socialMask
from copy import deepcopy
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
# from envs.ElevatorENV import Lift

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

# env = GatheringEnv(2)  # gym.make('CartPole-v1')
env = HarvestEnv(num_agents = 2)
# env.seed(args.seed)
# torch.manual_seed(args.seed)

agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device}
# agentParam =

model_name = "pg_social"
file_name = "/Users/xue/Desktop/Social Law/saved_weight/" + model_name
save_eps = 10
ifsave_model = True
# logger = Logger('./logs5')


class Agents():
    def __init__(self,agents,exploration=0.5):
        self.num_agent = len(agents)
        self.agents = agents
        self.exploration = exploration
        self.epsilon = 0.95


    def choose_action(self,state):
        actions = {}
        agentID = list(state.keys())
        i = 0
        for agent, s in zip(self.agents, state.values()):
            actions[agentID[i]] = int(agent.choose_action(s/255.).detach().numpy())
            i += 1
        return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, list(state), list(reward), list(state_), list(action)):
            agent.update(s/255.,r,s_/255.,a)

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
    multiPG = Agents([IAC(8,675,CNN=True,width=15,height=15,channel=3) for i in range(n_agents)])  # create PGagents as well as a social agent
    # multiPG = Social_Agents([social_IAC(8,400,agentParam) for i in range(n_agents)],agentParam)
    for i_episode in range(101):
        n_state, ep_reward = env.reset(), 0  # reset the env
        for t in range(1, 100):
            actions = multiPG.choose_action(n_state)  # agent.select_action(state)   #select masked actions for every agent
            # actions = multiPG.select_masked_actions(n_state)
            n_state_, n_reward, _, _ = env.step(actions)  # interact with the env
            if args.render:  # render or not
                env.render()
            # plt.close()
            # multiPG.push_reward(n_reward)  # each agent receive their own reward, the law receive the summed reward
            ep_reward += sum(n_reward.values())  # record the total reward
            multiPG.update(n_state.values(), n_reward.values(), n_state_.values(), actions.values())
            # multiPG.update_law()
            n_state = n_state_

        running_reward = ep_reward
        # loss = multiPG.update_agents()  # update the policy for each PGagent
        # multiPG.update_law()  # update the policy of law
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            # logger.scalar_summary("ep_reward", ep_reward, i_episode)

        # if i_episode % save_eps == 0 and i_episode > 1 and ifsave_model:
            # multiPG.save(file_name)
        #


if __name__ == '__main__':
    main()
