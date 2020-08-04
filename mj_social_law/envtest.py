from envs.SocialDilemmaENV.social_dilemmas.envir.harvest import HarvestEnv
from envs.SocialDilemmaENV.social_dilemmas.envir.cleanup import CleanupEnv
import random
import os
from gather_env import GatheringEnv
from envs.SocialDilemmaENV.social_dilemmas.constants import HARVEST_MAP,HARVEST_MAP2,HARVEST_MAP3,HARVEST_MAP4,HARVEST_MAP5
from envs.ElevatorENV.Lift import Lift

MINI_CLEANUP_MAP = [
    '@@@@@@',
    '@ P  @',
    '@H BB@',
    '@R BB@',
    '@S BP@',
    '@@@@@@',
]
class envGather:
    def __init__(self,n_agents,map_name='default_small'):
        self.n_agents = n_agents
        self.env = GatheringEnv(n_agents)
    
    def step(self,actions):
        return

class envLift:
    def __init__(self,n_agents,height):
        self.n_agents = n_agents
        self.height = height
        self.world = Lift(n_agents,height)
    
    def reset(self):
        return self.world.reset()
    
    def step(self,actions):
        return self.world.step(actions)
        

class envSocialDilemma:
    def __init__(self,envtype,n_agents):
        self.n_agents = n_agents
        if envtype == "harvest":
            self.world = HarvestEnv(ascii_map=HARVEST_MAP2,num_agents=self.n_agents)
        else:
            self.world = CleanupEnv(ascii_map=MINI_CLEANUP_MAP, num_agents=self.n_agents)

    def step_linear(self,actions):
        ## input: [1,2,4] ...
        actions = self.transferData(actions,'a2')
        state,reward,done,info = self.world.step(actions)
        reward = self.transferData(reward,'r')
        state = self.transferData(state,'sc')
        state = [ sa.flatten() for sa in state ]
        return state,reward,done,info

    def step(self,actions):
        ## input: [1,2,4] ...
        ## to CNN 
        actions = self.transferData(actions,'a2')
        state,reward,done,info = self.world.step(actions)
        reward = self.transferData(reward,'r')
        state = self.transferData(state,'sc')
        state_2d = [ sa.T/255 for sa in state]
        return state_2d,reward,done,info

    def reset_linear(self):
        state = self.world.reset()
        state = self.transferData(state,'sc')
        state = [ sa.flatten() for sa in state ]
        return state

    def reset(self):
        state = self.world.reset()
        state = self.transferData(state,'sc')
        state_2d = [ sa.T/255 for sa in state]
        return state_2d

    def mkdir(self,path):
        folder = os.path.exists(path)
        if not folder:                   
            os.makedirs(path)            
            print("---  new folder...  ---")
            print("---  OK  ---")
    
    def render(self,path,id):
        self.mkdir(path)
        self.world.render(path+"/im"+str(id)+".png")

    def transferData(self,data,mode):
        def actionDict(actions):
            ## actions: list of action for each agent
            ## return dict of agents' action e.g. {agent-1:2,...}
            action_dict = {}
            for i in range(len(actions)):
                name = "agent-"+str(i)
                action_dict[name] = actions[i].item()
            return action_dict
        def transReward(reward_dict):
            ## sutible for reward and obs in AC
            rewards = []
            for key,value in reward_dict.items():
                rewards.append(value)
            return rewards
        def actionDict2(actions):
            ## actions: list of action for each agent
            ## return dict of agents' action e.g. {agent-1:2,...}
            action_dict = {}
            for i in range(len(actions)):
                name = "agent-"+str(i)
                action_dict[name] = actions[i]#.item()
            return action_dict
        def transPicture(pic_data):
            pics = []
            for key,value in pic_data.items():
                pics.append(value.T/255)
            return pics
        if mode == 'r':
            return transReward(data)
        elif mode == 'a':
            return actionDict(data)
        elif mode == 'ddpg_s':
            return transPicture(data)
        elif mode == 'sc':
            return transReward(data)
        elif mode == "a2":
            return actionDict2(data)
        else:
            return []

if __name__ == "__main__":
    world = envSocialDilemma("cleanup",2)
    actions = [ random.randint(0,7) for i in range(world.n_agents)]
    state,reward,_,_ = world.step_linear(actions)
    print(state[0].shape)
    print(reward)