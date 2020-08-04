from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
import numpy as np
FIRING_CLEANUP_MAP = [
    '@@@@@@',
    '@    @',
    '@HHP @',
    '@RH  @',
    '@H P @',
    '@@@@@@',
]
CLEANUP_VIEW_SIZE = 1

n_agents = 2
n_states = (CLEANUP_VIEW_SIZE*2+1)*(CLEANUP_VIEW_SIZE*2+1)*3
world = CleanupEnv(ascii_map=FIRING_CLEANUP_MAP, num_agents=2)
world.reset()
rand_action = np.random.randint(9, size=2)
obs, rew, dones, info, = world.step({'agent-0': rand_action[0],
                                      'agent-1': rand_action[1]})
for key,value in obs.items():
    value = value.flatten()
    obs[key] = value

def contactSta(stadict,mode):
    sta = []
    for key,value in stadict.items():
        if mode == 's':
            value = value.flatten()
        sta.append(value)
    return sta
print("obs.....",contactSta(obs,'s')) 
print("rew.....",contactSta(rew,'r'))