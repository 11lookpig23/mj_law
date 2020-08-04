import numpy as np
import sys 
import os
#print(os.getcwd())
import random
import copy
from envs.ElevatorENV.gridworld import Grid


class Lift():
    def __init__(self,agent_num,height):
        self.agent_num = agent_num
        self.height = height
        obs_dim = height*4+1
        self.action_range = [0, 1, 2]
        self.grid = []
        self.busy_n = {}

    def get_game_list(self):
        pass

    def get_rewards(self):
        pass

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = [False]*self.agent_num
        info_n = {"new_act":[]}

        grid1,busy_n,obs_n,true_act_n,reward_n = self.elevator.run_a_step(action_n)
        self.grid = grid1
        self.busy_n = busy_n
        info_n["new_act"] = true_act_n
        ## change grid
        ## new action
        reward_x = [int(x) for x in reward_n ]
        return obs_n, reward_x, done_n, info_n
    def trans_obs(self,pos):
        shape =  self.grid.shape[0]*self.grid.shape[1]
        selfpos = np.zeros((self.grid.shape[0],self.grid.shape[1]))
        selfpos[pos] = 1
        selfpos = np.reshape(selfpos,shape)
        grid1 = copy.deepcopy(self.grid)
        grid1[pos] = 0
        otherpos = np.reshape(grid1,shape)
        inputarr = np.hstack((selfpos,otherpos))
        return inputarr
    def reset(self):
        ## obser_n
        ## 只是部分的obs
        arr = np.zeros(self.height*2)
        num = self.agent_num#self.height-3
        inx=random.sample(range(0,self.height*2),num)
        arr[inx] = 1
        arr = np.reshape(arr,(self.height,2))
        self.grid = arr
        obs_n = []
        namelist = []
        k2 = 1
        for i in range(self.height):
            for j in range(2):
                if self.grid[i,j]!=0:
                    inputarr0 = self.trans_obs((i,j))
                    if(k2%2==0):
                        busy = 0
                    else:
                        busy = 1
                    self.busy_n["Ag"+str(k2)] = busy
                    obs_n.append(np.hstack((inputarr0, np.array([busy]))))
                    namelist.append("Ag"+str(k2))
                    k2+=1
        ### create a grid world
        ## combine grid and busy

        self.elevator = Grid(self.grid,self.busy_n,self.agent_num,namelist)
        return obs_n
    
    def terminate(self):
        pass

    def render(self):
        print("ohhhhhhh")


def rule_agent(height,obs):
    ## obs: self-pos + other-pos + busy
    s1 = obs[:height*2]
    s2 = obs[height*2:4*height]
    busy = obs[-1]
    selfpos = s1.reshape((height,2))
    otherpos = s2.reshape((height,2))
    grid  = selfpos+otherpos
    #x = self.pos[0]
    #y = self.pos[1]
    print( " grid ",grid)
    pos = np.where(selfpos==1)
    x = int(pos[0])
    y = int(pos[1])
    
    print(" pos ",x,y)
    
    if busy == 1:
        if x==len(grid)-1:
            action = 2
            return action,None,flag
        if (grid[x+1,y]==1 and  y == 1):
            action = 1
        else:
            action = 2
    else:
        if y==0:
            action = 1
        else:
            action = 0
    return action

def getPos(state,hei):
    busy = state[-1]
    mypos = state[:hei*2]
    mypos = mypos.reshape((hei,2))
    finalpos = np.where(mypos==1)
    return busy,finalpos

if __name__ == "__main__":
    lift = Lift(4,5)
    obs_n = lift.reset()
    #print(" obs_n1 ",obs_n)
    #print( " ...a... ")
    #act = rule_agent(5,obs_n[0])
    #print( " ...b... ")
    #act1 = rule_agent(5,obs_n[1])
    #act = rule_agent(5,obs_n[0])
    lift.elevator.printGrid()
    obs_n, reward_x, done_n, info_n = lift.step(np.array([0,0,0,0]))

    lift.elevator.printGrid()
    #act = rule_agent(5,obs_n[0])
    #print( " ...d... ")
    #act = rule_agent(5,obs_n[1])
    #obs_n, reward_x, done_n, info_n = lift.step(np.array([act,act1]))
    print("obs_n[0]....",obs_n[0])
    for obs in obs_n:
        busy,pos = getPos(obs,5)
        print("busy...",int(busy),"coordination...", pos[1][0])
        print("=====")
