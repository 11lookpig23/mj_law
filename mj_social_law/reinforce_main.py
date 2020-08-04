import argparse
import gym
import numpy as np
from itertools import count
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gather_env import GatheringEnv
from PGagent import PGagent,social_agent,newPG
from network import socialMask
from copy import deepcopy
import os
from multiAG import independentAgent,socialAgents
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

n_agents = 2
env = GatheringEnv(n_agents,"default_small2")
env.seed(args.seed)
torch.manual_seed(args.seed)

test_mode = False


names = {"social":"pg_social","base":"pg_indi","single":"pg_single_less","forbid":"base_forbid"}
mode = "base"
model_name = names[mode]
file_name  = "train_para/"+model_name
agentParam = {"gamma":args.gamma,"LR":0.005,"device":device,"ifload":False,"filename": file_name}
save_eps = 100

n_episode = 2001
n_steps = 500

if test_mode:
    ifsave_model = False
    ifsave_data = False
    render = True
else:
    ifsave_model = True
    ifsave_data = True
    render = False


def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def random_agent():
    all_rw = []
    for i_episode in range(n_episode):
        ep_rw = 0
        for t in range(n_steps):
            actions = [ random.randint(0,7) for i in range(n_agents)]
            n_state, n_reward, _, _ = env.step(actions)
            ep_rw+=sum(n_reward)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_rw, ep_rw))
        all_rw.append(ep_rw)
        np.save("data/"+"random1"+".npy",all_rw)


def main():
    #agent = PGagent(agentParam)
    all_rw = []
    #n_agents = 1#2
    if mode == "social":
        multiPG = socialAgents([newPG(add_para(i)) for i in range(n_agents)],agentParam)
    else:
        multiPG = independentAgent([PGagent(add_para(i)) for i in range(n_agents)])

    for i_episode in range(n_episode):
        n_state, ep_reward = env.reset(), 0
        for t in range(n_steps):
            if mode == "social":
                actions = multiPG.select_mask_actions(n_state)
            else:
                actions = multiPG.select_actions(n_state)##agent.select_action(state)
            #actions = [ random.randint(0,7) for i in range(n_agents)]
            n_state, n_reward, _, _ = env.step(actions)
            if render:
                env.render()
            multiPG.push_reward(n_reward)
            ep_reward += sum(n_reward)

        running_reward = ep_reward
        if test_mode == False:
            multiPG.update_agents()

        all_rw.append(ep_reward)
        if i_episode % (args.log_interval*2) == 0 and ifsave_data:
            np.save("data/"+model_name+".npy",all_rw)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if i_episode % save_eps == 0 and i_episode>10 and ifsave_model:
            multiPG.save(file_name)


if __name__ == '__main__':
    main()
    #random_agent()