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
from PGagent import PGagent, social_agent, newPG, IAC, social_IAC,Centralised_AC,Law_agent
from network import socialMask,Centralised_Critic
from copy import deepcopy
#from logger import Logger
from torch.utils.tensorboard import SummaryWriter
# from envs.ElevatorENV import Lift
from centCrt import CenAgents
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = GatheringEnv(2,"default_small2")#"mini_map")  # gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)

# agentParam =

model_name = "gathering_social_"#gathering_1"
file_name = "save_weight/" + model_name
ifload = False
save_eps = 20
ifsave_model = True
# logger = Logger('./logs5')
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name,}
n_episode = 101
n_steps = 1000
line = 10


class Agents():
    def __init__(self,agents):
        self.num_agent = len(agents)
        self.agents = agents

    def choose_action(self,state):
        actions = []
        #state = torch.Tensor(state).to(device)
        for agent, s in zip(self.agents, state):
            #s = torch.Tensor(s).to(device)
            actions.append(int(agent.choose_action(s).detach()))
        return actions
    def choose_masked_actions(self, state,mask_probs):
        actions = []
        for agent, s, rule in zip(self.agents, state,mask_probs):
            #s = torch.Tensor(s).to(device)
            actions.append(int(agent.choose_mask_action(s,rule).detach()))
        return actions
    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a)

    def save(self,file_name):
        for i,ag in zip(range(self.num_agent),self.agents):
            torch.save(ag.actor,file_name+"actor_"+str(i)+".pth")
            torch.save(ag.critic,file_name+"critic_"+str(i)+".pth")

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

    def save(self,file_name):
        torch.save(self.Law.policy,file_name+"pg_law"+".pth")
        for i,ag in zip(range(self.n_agents),self.agents):
            torch.save(ag.actor,file_name+"actor_"+str(i)+".pth")
            torch.save(ag.critic,file_name+"critic_"+str(i)+".pth")

def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def main():
    # agent = PGagent(agentParam)
    writer = SummaryWriter('runs/iac_'+model_name)
    n_agents = 2
    state_dim = 400
    action_dim = 8
    # multiPG = independentAgent([PGagent(agentParam) for i in range(n_agents)])
    multiPGCen = CenAgents([Centralised_AC(8,state_dim,add_para(i),useLaw=False) for i in range(n_agents)],state_dim,agentParam)  # create PGagents as well as a social agent
    #multiPG = Law_agent(action_dim,state_dim,agentParam,n_agents)
    multiPG = Agents([IAC(8,400,add_para(i),useLaw=True) for i in range(n_agents)])  # create PGagents as well as a social agent
    #multiPG = Social_Agents([social_IAC(8,400,agentParam) for i in range(n_agents)],agentParam)
    for i_episode in range(n_episode):
        #print(" =====================  ")
        n_state, ep_reward = env.reset(), 0  # reset the env
        for t in range(n_steps):
            #print(" =====================  ",n_state)
            if (int(i_episode/line)%2==True):#i_episode<line:
                actions = multiPGCen.choose_action(n_state)
            else:
                mask_probs = multiPGCen.choose_mask_probs(n_state)
                actions = multiPG.choose_masked_actions(n_state,mask_probs)  # agent.select_action(state)   #select masked actions for every agent
            # actions = multiPG.select_masked_actions(n_state)
            n_state_, n_reward, _, _ = env.step(actions)  # interact with the env
            if args.render and i_episode%50==0:  # render or not
                env.render()
            # multiPG.push_reward(n_reward)  # each agent receive their own reward, the law receive the summed reward
            ep_reward += sum(n_reward)  # record the total reward
            if (int(i_episode/line)%2==True):
                multiPGCen.update(n_state, n_reward, n_state_, actions)
            else:
                multiPG.update(n_state, n_reward, n_state_, actions)
            # multiPG.update_law()
            n_state = n_state_

        running_reward = ep_reward
        # loss = multiPG.update_agents()  # update the policy for each PGagent
        # multiPG.update_law()  # update the policy of law
        writer.add_scalar("ep_reward", ep_reward, i_episode)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            # logger.scalar_summary("ep_reward", ep_reward, i_episode)
        if i_episode % save_eps == 0 and i_episode > 11 and ifsave_model:
            multiPG.save(file_name)
            #pass


if __name__ == '__main__':
    main()
