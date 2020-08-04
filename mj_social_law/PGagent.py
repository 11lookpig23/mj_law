import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from network import Policy,socialMask,Actor,Critic,CNN_preprocess,Centralised_Critic,ActorLaw
import copy
import itertools
import random
import torchsnooper
class PGagent():
    def __init__(self,agentParam):
        self.state_dim = 400#env.observation_space.shape[0]
        self.action_dim = 8#env.action_space.n
        self.gamma = agentParam["gamma"]
        # init N Monte Carlo transitions in one game
        self.saved_log_probs = []
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.rewards = []
        self.device = agentParam["device"]
        # init network parameters
        self.policy = Policy(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=agentParam["LR"])
        self.eps = np.finfo(np.float32).eps.item()

        # init some parameters
        self.time_step = 0


    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item()


    def update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device).type(self.FloatTensor)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss)
        policy_loss = policy_loss.sum()
        temp = copy.copy(policy_loss)
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return temp


class social_agent(PGagent):
    def __init__(self,agentParam):
        super().__init__(agentParam)
        self.policy = socialMask(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        #    #[[pi_1,pi_2,...],[pi_1,pi_2,...],...
        self.pi_step = []
        #    #[[prob_1,prob_2,...],[prob_1,prob_2,...],....
        self.prob_social = []
        # self.reward = [sum(R1),sum(R2),....]
    def select_action(self,state):                      #select
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        #self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),probs

    def maskFunc(self,probs,masks):
        return F.softmax(torch.mul(probs,masks))
    def update(self,n_agents):
        R = 0
        policy_loss = []
        returns_sum = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns_sum.insert(0, R)
        returns = [ [r]*n_agents for r in returns_sum]
        self.saved_log_probs = []
        for k in range(len(self.pi_step)):                       #pi_step is the list of unmasked policy(prob ditribution) for each agent
            pi = self.pi_step[k].detach()
            new_probs = self.maskFunc(pi,self.prob_social[k])    #prob_social is the list of masks for each agent
            m = Categorical(new_probs)
            action = m.sample()                                  #sample from the distribution
            self.saved_log_probs.append(m.log_prob(action).to(self.device)) #save the logged prob for sampled action
        returns = np.array(returns).flatten()
        returns = torch.tensor(returns).to(self.device).type(self.FloatTensor)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):   #calculate the -log(pi)R
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()                                   #back propagation
        self.optimizer.step()
        del self.rewards[:]
        del self.pi_step[:]
        del self.prob_social[:]


class newPG(PGagent):
    def __init__(self,agentParam):
        super().__init__(agentParam)
    def maskFunc(self,probs,masks):
        temp = torch.mul(probs,masks)
        # temp[0,0] = 1
        # for i in range(1,8):
        #     temp[0,i] = 0
        return F.softmax(temp)
    def select_masked_action(self,state,masks):
        masks = masks.detach()
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        new_probs = self.maskFunc(probs,masks)
        m = Categorical(new_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),probs

# class Actor(nn.Module):
#     def __init__(self,action_dim,state_dim):
#         super(Actor,self).__init__()
#         self.Linear1 = nn.Linear(state_dim,128)
#         # self.Dropout1 = nn.Dropout(p=0.3)
#         self.Linear2 = nn.Linear(128,action_dim)
#
#     def forward(self,x):
#         x = self.Linear1(x)
#         # x = self.Dropout1(x)
#         x = F.relu(x)
#         x = self.Linear2(x)
#         return F.softmax(x)
#
# class Critic(nn.Module):
#     def __init__(self,state_dim):
#         super(Critic,self).__init__()
#         self.Linear1 = nn.Linear(state_dim, 128)
#         # self.Dropout1 = nn.Dropout(p=0.3)
#         self.Linear2 = nn.Linear(128, 1)
#
#     def forward(self,x):
#         x = self.Linear1(x)
#         # x = self.Dropout1(x)
#         x = F.relu(x)
#         x = self.Linear2(x)
#         return x

class IAC():
    def __init__(self,action_dim,state_dim,agentParam,useLaw,CNN=False, width=None, height=None, channel=None):
        self.CNN = CNN
        self.device = agentParam["device"]
        if CNN:
            self.CNN_preprocessA = CNN_preprocess(width,height,channel)
            self.CNN_preprocessC = CNN_preprocess(width,height,channel)
            state_dim = self.CNN_preprocessA.get_state_dim()
        #if agentParam["ifload"]:
            #self.actor = torch.load(agentParam["filename"]+"actor_"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
            #self.critic = torch.load(agentParam["filename"]+"critic_"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
        #else:
        if useLaw:
            self.actor = ActorLaw(action_dim,state_dim).to(self.device)
        else:
            self.actor = Actor(action_dim,state_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.noise_epsilon = 0.99
        self.constant_decay = 0.1
        self.optimizerA = torch.optim.Adam(self.actor.parameters(), lr = 0.001)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(), lr = 0.001)
        self.lr_scheduler = {"optA":torch.optim.lr_scheduler.StepLR(self.optimizerA,step_size=1000,gamma=0.9,last_epoch=-1),
                             "optC":torch.optim.lr_scheduler.StepLR(self.optimizerC,step_size=1000,gamma=0.9,last_epoch=-1)}
        if CNN:
            # self.CNN_preprocessA = CNN_preprocess(width,height,channel)
            # self.CNN_preprocessC = CNN_preprocess
            self.optimizerA = torch.optim.Adam(itertools.chain(self.CNN_preprocessA.parameters(),self.actor.parameters()),lr=0.0001)
            self.optimizerC = torch.optim.Adam(itertools.chain(self.CNN_preprocessC.parameters(),self.critic.parameters()),lr=0.001)
            self.lr_scheduler = {"optA": torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=10000, gamma=0.9, last_epoch=-1),
                                 "optC": torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=10000, gamma=0.9, last_epoch=-1)}
        # self.act_prob
        # self.act_log_prob
    #@torchsnooper.snoop()
    def choose_action(self,s):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1,3,15,15)))
        self.act_prob = self.actor(s) + torch.abs(torch.randn(self.action_dim)*0.05*self.constant_decay).to(self.device)
        self.constant_decay = self.constant_decay*self.noise_epsilon
        self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def choose_act_prob(self,s):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        self.act_prob = self.actor(s,[],False)
        return self.act_prob.detach()

    def choose_mask_action_indi(self,s,pi):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1,3,15,15)))
        self.act_prob = self.actor(s,pi,True) + torch.abs(torch.randn(self.action_dim)*0.05*self.constant_decay).to(self.device)
        self.constant_decay = self.constant_decay*self.noise_epsilon
        self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def choose_mask_action(self,s,rule):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1,3,15,15)))
        self.act_prob = self.actor(s,rule,True) + torch.abs(torch.randn(self.action_dim)*0.05*self.constant_decay).to(self.device)
        self.constant_decay = self.constant_decay*self.noise_epsilon
        self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp
    def cal_tderr(self,s,r,s_,A_or_C=None):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        s_ = torch.Tensor(s_).unsqueeze(0).to(self.device)
        if self.CNN:
            if A_or_C == 'A':
                s = self.CNN_preprocessA(s.reshape(1,3,15,15))
                s_ = self.CNN_preprocessA(s_.reshape(1,3,15,15))
            else:
                s = self.CNN_preprocessC(s.reshape(1,3,15,15))
                s_ = self.CNN_preprocessC(s_.reshape(1,3,15,15))
        v_ = self.critic(s_).detach()
        v = self.critic(s)
        return r + 0.9*v_ - v

    def learnCritic(self,s,r,s_):
        td_err = self.cal_tderr(s,r,s_)
        loss = torch.mul(td_err,td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()
    #@torchsnooper.snoop()
    def learnActor(self,s,r,s_,a):
        td_err = self.cal_tderr(s,r,s_)
        #print("self.act_prob[0][a] ..... ",self.act_prob[0][a])
        m = torch.log(self.act_prob[0][a])
        #print("m ............  ",m)
        temp = m*td_err.detach()
        #print("td_err .........",td_err.detach())
        #print("temp ..... ",temp)
        loss = -torch.mean(temp)
        #print("loss ..... ",loss)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self,s,r,s_,a):
        self.learnCritic(s,r,s_)
        self.learnActor(s,r,s_,a)

class Law_agent(IAC): 
    def __init__(self,action_dim,state_dim,agentParam,num_agent):
        self.num_agent = num_agent
        super().__init__(action_dim,state_dim,agentParam)
        if agentParam["ifload"]:
            self.critic = torch.load(agentParam["filename"]+"cent_critic_"+".pth",map_location = torch.device('cuda'))
        else:
            self.critic = Centralised_Critic(state_dim,self.num_agent).to(self.device)
    
    def choose_action(self,state):
        state = torch.Tensor(state).to(self.device)
        self.act_prob = self.actor(state) + torch.abs(torch.randn(self.num_agent,self.action_dim)*0.05*self.constant_decay).to(self.device)
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def td_err(self, s, r, s_):
        s = torch.Tensor(s).reshape((1,-1)).unsqueeze(0).to(self.device)
        s_ = torch.Tensor(s_).reshape((1,-1)).unsqueeze(0).to(self.device)
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
        self.lr_scheduler["optC"].step()
    
    #@torchsnooper.snoop()
    def LearnLaw(self,actions,td_err):
        for i in range(1):
            m =  torch.log(self.act_prob[i,actions[i]])
            temp = m*(td_err.detach()).to(self.device)
            loss = -torch.mean(temp)
            self.optimizerA.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizerA.step()
            self.lr_scheduler["optA"].step()


    def update(self, state, reward, state_, action):
        td_err = self.td_err(state,sum(reward),state_)
        self.LearnLaw(action,td_err)
        self.LearnCenCritic(state,sum(reward),state_)    
    
    def save(self,file_name):
        torch.save(self.actor,file_name+"actor_social"+".pth")
        torch.save(self.critic,file_name+"critic_social"+".pth")

class Centralised_AC(IAC):
    def __init__(self,action_dim,state_dim,agentParam,useLaw):
        super().__init__(action_dim,state_dim,agentParam,useLaw)
        self.critic = None
        if agentParam["ifload"]:
            self.actor = torch.load(agentParam["filename"]+"actor_"+agentParam["id"]+".pth",map_location = torch.device('cuda'))

    # def cal_tderr(self,s,r,s_):
    #     s = torch.Tensor(s).unsqueeze(0)
    #     s_ = torch.Tensor(s_).unsqueeze(0)
    #     v = self.critic(s).detach()
    #     v_ = self.critic(s_).detach()
    #     return r + v_ - v

    def learnActor(self,a,td_err):
        m = torch.log(self.act_prob[0][a]).to(self.device)
        temp = m*(td_err.detach()).to(self.device)
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self,s,r,s_,a,td_err):
        self.learnActor(a,td_err)

class social_IAC(IAC):
    def __init__(self,action_dim,state_dim,agentParam):
        super().__init__(action_dim,state_dim)
        self.saved_log_probs = []
        self.device = agentParam["device"]
        self.reward = []
        self.rewards = []

    def select_masked_action(self,state,masks):
        masks = masks.detach()
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.act_prob = self.actor(state)[0]
        new_probs = self.maskFunc(self.act_prob.detach(),masks)
        m = Categorical(new_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),self.act_prob

    def maskFunc(self,prob,mask):
        return F.softmax(torch.mul(prob,mask))
