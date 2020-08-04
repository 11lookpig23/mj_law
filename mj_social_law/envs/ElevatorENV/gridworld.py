import numpy as np
import copy
import random

class Agent():
    def __init__(self,id,pos,empty,busy):
        self.id = id
        self.pos = pos
        self.busy = busy
        self.empty = empty
        self.step = 1
        self.finished = False
        self.block = False
        self.fix = False
        self.move = 0

    def reset(self):
        ## 出电梯则reset, reset 保持坐标不变，empty = True，其他清空，等待被移动到第一行,id为none，移动至第一行时如果是agnet,重新赋予id,在上升过程中id不变
        ## 用于none 或其他reset
        self.empty = True
        self.finished = False
        self.id = 'none'
        self.block = False
        self.fix = False
        self.move = 0

    def birth(self,busy):
        self.empty = False
        #self.id = id
        self.busy = busy
        self.step = 0
        self.block = False
        self.fix = False
        self.pos = [-1,-1]
        self.finished = False
        #self.move = 0
    
    def set_pos(self,pos):
        self.pos = pos

    def next_step(self,grid,height):
    ## 同一行，先看up，再看switch
    ## 如果不空，那就move，否则就move = 0
    ## 如果移动，则改变pos,reward
    ## 如果向前一格是顶点,...
        newpos,onereward = self.newPos(height)
        #print("newpos===",newpos)
        if self.finished == True:
            if self.busy == 1: ## 1 is busy
                rw = 0 #-2*self.step
            else:
                rw = -1
            ### --- ??
            ## self.reset()
            return rw,self.finished,self.move
        if grid[newpos[0],newpos[1]].empty:
            #print("empty####")
            self.pos = newpos
            return onereward, self.finished, self.move
        else:
            self.block = True
            if self.busy == 0:
                r1 = 0 ##  不忙
            else:
                r1 = -1
            return r1, self.finished, 0
    
    def newPos(self,height):
        if self.busy == 0:
            r1 = 0 ##  不忙
            r2 = -1
        else:
            r1 = -1
            r2 = 0
        if self.move == 0:
            return self.pos,r1
        elif self.move == 1:
            return [self.pos[0],1-self.pos[1]],r1
        else:
            if self.pos[0]== height-1:
                self.finished = True
                ## 需要改一下
                return [self.pos[0]+1,self.pos[1]],r2
            else:
                return [self.pos[0]+1,self.pos[1]],r2


class Grid():
    def __init__(self,grid,busy_n,num,namelist):
        self.height = grid.shape[0]
        self.world = np.full((self.height,2),'a',dtype = object)
        self.waitList = []
        self.namelist = namelist
        self.record = {}
        self.busy_n =  busy_n
        self.num = num
        k = 0
        k2 = 1
        for i in range(self.height):
            for j in range(2):
                name='agent'+str(k)
                if grid[i,j] == 0:
                    locals()[name]= Agent("none",[i,j],True,0)
                else:
                    locals()[name]= Agent(namelist[k2-1],[i,j],False,busy_n[namelist[k2-1]])
                    k2+=1
                self.world[i][j] = locals()[name]
                k += 1

    def escaUP(self):
        for j in range(2):
            ag = self.world[self.height-1,j]
            if ag.id!='none':
                busy = int(1-ag.busy)
                ag.birth(busy = busy)
                self.busy_n[ag.id] = ag.busy
                self.waitList.append(copy.deepcopy(ag))
                #self.waitList[-1].pg = tmp_pg
            else:
                ag.reset()
        #[a1,a2] = [self.world[self.height-1,0],self.world[self.height-1,1]]
        self.world[1:self.height] = self.world[:self.height-1]
        #for j in range(2):
        if len(self.waitList)==1:
            self.world[0][0] = self.waitList[0]
            del self.waitList[0]
            self.world[0][1] = Agent("none",[0,1],True,0)
        elif len(self.waitList)==0:
            self.world[0][1] = Agent("none",[0,1],True,0)
            self.world[0][0] = Agent("none",[0,0],True,0)
        else:
            self.world[0][0] = self.waitList[0]
            self.world[0][1] = self.waitList[1]
            del self.waitList[0]
            del self.waitList[0]
        for i in range(self.height):
            for j in range(2):
                self.world[i][j].set_pos([i,j])
                self.world[i][j].step+=1

    def getOrder(self,arr):
        # arr = [agent1,agent2]
        #if (arr[0].move == 2 and arr[1].move == 2):
        #    return random.choice((0,1))
        ## [2,0] 先 2
        ## [2,1] 先 2
        ## [2,2] 先 2
        ## [1,1] 先 ?? 随便 ，因为动不了
        ## [1,none] 先 none
        ## [1,0] 随便
        ## [0,0] 随便 ，因为动不了
        if arr[0].move == 2:
            return 0
        elif arr[1].move == 2:
            return 1
        elif (arr[1].move == 1 and arr[0].id == "none"):
            return 0
        elif (arr[1].id == "none" and arr[0].move == 1):
            return 1
        else:
            return random.choice((0,1))

    def takenext(self,k,j):
        #inputarr,action,probas,flag = #self.world[k,j].predict(self.worldtomatrix())
        rw, done, true_act = self.world[k,j].next_step(self.world,self.height)
        self.world[k,j].move = true_act
        self.record[self.world[k,j].id] = [rw,true_act]
        ## 跳脱
        flag = " no "
        if self.world[k,j].finished == True:
            flag = flag+" finish"
            busy = int(1-self.world[k,j].busy)
            self.world[k,j].birth(busy = busy)
            self.busy_n[self.world[k,j].id] = self.world[k,j].busy
            self.waitList.append(copy.deepcopy(self.world[k,j]))
            self.world[k,j].reset()
            return flag
        newpos = self.world[k,j].pos
        if (self.world[k,j].block!=True and self.world[k,j].move!=0 and self.world[k,j].finished == False and self.world[k,j].empty == False and self.world[k,j].id!="none" ):
            flag = flag+" exeut"
            tmp = copy.deepcopy(self.world[k,j])
            tmp0 = copy.deepcopy(self.world[tuple(newpos)])
            self.world[k,j] = tmp0
            self.world[tuple(newpos)] = tmp
        self.world[k,j].block = False
        return flag


    def take_action(self):
        for k in range(self.height-1,-1,-1):
            arr = self.world[k]
            j = self.getOrder(arr)
            tic = 0
            if (self.world[k,j].id!="none" and self.world[k,1-j].id == "none"):
                tic = 1
            if self.world[k,j].empty == False:
                flag =  self.takenext(k,j)
            if self.world[k,1-j].empty == False:
                if tic!=1:
                    flag = self.takenext(k,1-j)
                else:
                    flag = "nonono"

    def transAction(self,action,pos,agent):
        ## stay: 0; switch: 1; up: 2
        ## left: 0; right: 1; up: 2
        ## transA_B: 
        '''
        now_left/right to l/r: left/right --> stay
        left/right to r/l: switch
        up: up
        '''
        if action == 0:
            if pos[1] == 0:
                return 0
            else:
                return 1
        elif action == 1:
            if pos[1] == 0:
                return 1
            else:
                return 0
        return 2

    def set_action(self,action_n):
        for i in range(self.height):
            for j in range(2):
                if self.world[i,j].id!="none":
                    ### 还有waitList .... 
                    preACT = action_n[self.namelist.index(self.world[i,j].id)]
                    newACT = self.transAction(preACT,(i,j),self.world[i,j])
                    self.world[i,j].move = newACT#action_n[self.namelist.index(self.world[i,j].id)]
        for w in self.waitList:
            if w.busy == 0:
                r1 = 0 ##  不忙
            else:
                r1 = -1
            w.move = 0
            ### 改进否？？
            self.record[w.id][1] = w.move
            self.record[w.id][0] = r1
            #print(" waitlist ==== info ",w.id," ++++ ",self.record[w.id][1])

    def gridtomatrix(self):
        mat = np.zeros((self.height,2))
        for i in range(self.height):
            for j in range(2):
                if self.world[i,j].id == "none":
                     mat[i,j] = 0
                else:
                    mat[i,j] = 1
        return mat

    def trans_obs(self,pos,mat):
        shape = mat.shape[0]*mat.shape[1]
        selfpos = np.zeros((mat.shape[0],mat.shape[1]))
        selfpos[pos] = 1
        if pos == [-1,-1]:
            selfpos = np.zeros((mat.shape[0],mat.shape[1]))
        selfpos = np.reshape(selfpos,shape)
        grid1 = copy.deepcopy(mat)
        if pos == [-1,-1]:
            otherpos = np.reshape(grid1,shape)
        else:
            grid1[pos] = 0
            otherpos = np.reshape(grid1,shape)
        inputarr = np.hstack((selfpos,otherpos))
        return inputarr


    def printGrid(self):
        def cell(tup):
            if tup.id == 'none':
                return [ tup.id, tup.empty, "xxxxx", "xxxxx" , "xx" ]
            else:
                actions = ["stay ","switch","upup "]
                busys = ["noBusy","isBusy"]
                return [tup.id, tup.empty, actions[int(tup.move)], busys[tup.busy] ,str(tup.step)+'s']
        for j in range(self.height-1,-1,-1):
            print("| " ,cell(self.world[j,0]),"  |  ",cell(self.world[j,1]))
        print("+++++++++++")

    def run_a_step(self,action_n):
        self.set_action(action_n)
        #print(" +++++++ UP UP +++++ ")
        #self.printGrid()
        self.take_action()
        #print( " ******* take action ******** ")
        #self.printGrid()
        self.escaUP()
       # print(" +++++++ after UP UP +++++ ")
       # self.printGrid()
        true_act_n = []
        reward_n = []
        obs_n = []
        obsdict = {}
        mat = self.gridtomatrix()
        for i in range(self.height):
            for j in range(2):
                if self.world[i,j].id != "none":
                    inputarr0 = self.trans_obs((i,j),mat)
                    inputarr1 = np.hstack((inputarr0,np.array([self.busy_n[self.world[i,j].id]])))
                    obsdict[self.world[i,j].id] = inputarr1
        for key in self.namelist:
            true_act_n.append(self.record[key][1])#np.array(self.record[key][1]))   
            reward_n.append(self.record[key][0])#np.array(self.record[key][0]))
            try:
                obs_n.append(obsdict[key])
            except KeyError:
                #print("obs -- error ")
                for w in self.waitList:
                    if w.id == key:
                        obs_key = np.hstack(( self.trans_obs([-1,-1],mat) ,np.array([w.busy])))
                        obs_n.append(obs_key)
        if (len(true_act_n)!=self.num or len(reward_n)!=self.num or len(obs_n)!=self.num):
            print(" length ----- ERROR -------------- ")
        
        return mat,self.busy_n,obs_n,true_act_n,reward_n #np.asarray(true_act_n)
