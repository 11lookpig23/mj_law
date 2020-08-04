
import numpy as np
from itertools import count
from network import socialMask,Centralised_Critic,Actor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CenAgents():
    def __init__(self,agents,state_dim,agentParam):
        self.num_agent = len(agents)
        self.agents = agents
        if agentParam["ifload"]:
            self.critic = torch.load(agentParam["filename"]+"cent_critic_"+".pth",map_location = torch.device('cuda'))
        else:
            self.critic = Centralised_Critic(state_dim,self.num_agent).to(device)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(),lr=0.01)
        self.lr_schedulerC = torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=1000, gamma=0.9, last_epoch=-1)
        for i in self.agents:
            i.critic = self.critic

    def choose_action(self,state):
        actions = []
        for agent, s in zip(self.agents, state):
            actions.append(int(agent.choose_action(s).detach()))
        
        #for s in state:
        #    actions.append(int(self.agents[1].choose_action(s).detach()))
            #actions.append(int(self.agents[1].choose_action(state[1]).detach()))
        return actions
    def choose_masked_actions(self, state,pi_probs):
        actions = []
        for agent, s, rule in zip(self.agents, state,pi_probs):
            #s = torch.Tensor(s).to(device)
            actions.append(int(agent.choose_mask_action_indi(s,rule).detach()))
        return actions
    def choose_mask_probs(self,state):
        probs = []
        for agent, s in zip(self.agents, state):
            probs.append(agent.choose_act_prob(s))
        return probs

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
    
    
    def hard_copy(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update_share(self, state, reward, state_, action):
        td_err = self.td_err(state,sum(reward),state_)
        for i,agent, s, r, s_,a in zip(range(self.num_agent),self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a,td_err)
            if i<self.num_agent-1:
                self.hard_copy(self.agents[i+1].actor,agent.actor)
            else:
                self.hard_copy(self.agents[0].actor,agent.actor)
        #self.agents[1].update(state[1], reward[1], state_[1], action[1], td_err)
        self.LearnCenCritic(state,sum(reward),state_)

    def update(self, state, reward, state_, action):
        td_err = self.td_err(state,sum(reward),state_)
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a,td_err)
        #self.agents[1].update(state[1], reward[1], state_[1], action[1], td_err)
        self.LearnCenCritic(state,sum(reward),state_)

    def save(self,file_name):
        for i,ag in zip(range(self.num_agent),self.agents):
            torch.save(ag.actor,file_name+"actor_"+str(i)+".pth")
        torch.save(self.critic,file_name+"cent_critic_"+".pth")


#   centCritic (cent)
#   rulePolicy(dcent)
