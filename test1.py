import gym
import time
import numpy as np
import random
#from utils.pong_utils import *
from train import make_env



np.random.seed(1)
random.seed(1)
render=False
#env = gym.make('Pong-v0')
env,_ = make_env('Pong')

n= env.action_space.n
s_prev = env.reset()
score=0
while True:        #env.render()
   # print(observation)
    if render:
        env.render()
    action = env.action_space.sample()
    s, r, done, info = env.step(action)
    #ball=ball_position(s)
    #I=preprocess_screen(s)

    #print(  'ball={}, crit={}'.format(ball,comp_crit_lin(s,s_prev))  )
    score+=r
    s_prev=s

    if done:
        print("score={}".format(score))
        s_prev = env.reset()
        score=0
        #break