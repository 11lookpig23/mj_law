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
from PGagent import PGagent, social_agent, newPG, IAC, social_IAC,Centralised_AC
from network import socialMask,Centralised_Critic
from copy import deepcopy
#from logger import Logger
from envtest import envSocialDilemma,envLift
from torch.utils.tensorboard import SummaryWriter
# from envs.ElevatorENV import Lift

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

n_agents = 5
height = 5
env = envLift(n_agents,height)
torch.manual_seed(args.seed)

# agentParam =

#model_name = "lift_1"
model_name = "lift_cenTr"

file_name = "save_weight/" + model_name
ifload = True
save_eps = 50
ifsave_model = True
# logger = Logger('./logs5')
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name,}
n_episode = 301
n_steps = 200#1000
class CenAgents():
    def __init__(self,agents,state_dim,agentParam):
        self.num_agent = len(agents)
        self.agents = agents
        if agentParam["ifload"]:
            self.critic = torch.load(agentParam["filename"]+"cent_critic_"+".pth",map_location = torch.device('cuda'))
        else:
            self.critic = Centralised_Critic(state_dim,self.num_agent)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(),lr=0.01)
        self.lr_schedulerC = torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=1000, gamma=0.9, last_epoch=-1)
        for i in self.agents:
            i.critic = self.critic

    def choose_action(self,state):
        actions = []
        for agent, s in zip(self.agents, state):
            actions.append(int(agent.choose_action(s).detach()))
        return actions

    def td_err(self, s, r, s_):
        s = torch.Tensor(s).reshape((1,-1)).unsqueeze(0).to(device)
        s_ = torch.Tensor(s_).reshape((1,-1)).unsqueeze(0).to(device)
        v = self.critic(s)
        v_ = self.critic(s_).detach()
        return r + 0.9*v_ - v

    def LearnCenCritic(self, s, r, s_):
        td_err = self.td_err(s,r,s_)
        # m = torch.log(self.agents.act_prob[a[0]]*self.agents.act_prob[a[1]])
        loss = torch.mul(td_err,td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_schedulerC.step()

    def update(self, state, reward, state_, action):
        td_err = self.td_err(state,sum(reward),state_)
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a,td_err)
        self.LearnCenCritic(state,sum(reward),state_)
    def save(self,file_name):
        for i,ag in zip(range(self.num_agent),self.agents):
            torch.save(ag.actor,file_name+"actor_"+str(i)+".pth")
        torch.save(self.critic,file_name+"cent_critic_"+".pth")

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
    # [height*4+1,3]
    # agent = PGagent(agentParam)
    state_dim = height*4+1
    writer = SummaryWriter('runs/iac_'+model_name)
    # multiPG = independentAgent([PGagent(agentParam) for i in range(n_agents)])
    #multiPG = Agents([IAC(3,height*4+1,add_para(i)) for i in range(n_agents)])  # create PGagents as well as a social agent
    # multiPG = Social_Agents([social_IAC(8,400,agentParam) for i in range(n_agents)],agentParam)
    multiPG = CenAgents([Centralised_AC(3,state_dim,add_para(i)) for i in range(n_agents)],state_dim,agentParam)  # create PGagents as well as a social agent
    for i_episode in range(100,n_episode):
        #print(" =====================  ")
        n_state, ep_reward = env.reset(), 0  # reset the env
        for t in range(n_steps):
            #print(" =====================  ")

            actions = multiPG.choose_action(n_state)  # agent.select_action(state)   #select masked actions for every agent
            # actions = multiPG.select_masked_actions(n_state)
            n_state_, n_reward, _, _ = env.step(actions)  # interact with the env
            #if args.render and i_episode%20==0:  # render or not
            #    env.render()
            # multiPG.push_reward(n_reward)  # each agent receive their own reward, the law receive the summed reward
            ep_reward += sum(n_reward)  # record the total reward
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
        if i_episode % save_eps == 0 and i_episode > 15 and ifsave_model:
            multiPG.save(file_name)
        #


if __name__ == '__main__':
    main()
