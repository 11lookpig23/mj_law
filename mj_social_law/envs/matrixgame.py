import numpy as np
import sys 
#sys.path.append("..")#/..") 

import random
import copy

class MatrixGame():
    def __init__(self):
        self.agent_num = 2
        self.payoff = ([[3,0],[5,1]],
                        [[3, 5],[0,1]])
        self.action_range = [0,1]


    def step(self,action_n):
        obs_n = [[0],[0]]
        reward_n = [self.payoff[0][action_n[0]][action_n[1]],self.payoff[1][action_n[0]][action_n[1]]]
        done_n = [False]*2
        info_n = {}
        return obs_n, reward_n, done_n, info_n
        
    def reset(self):
        return [[0],[0]]

if __name__ == '__main__':
    game = MatrixGame()
    print(game.step([1,0]))
