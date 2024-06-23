import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym
from collections import deque
import torch.optim as optim
import math
import random

device ='cuda' if torch.cuda.is_available() else 'cpu'

def train(DQN,episodes):
  # hyperparameters
  max_iter = 200
  eps = 1.0
  eps_decay = 1000
  Memory_size = 1000
  batch_size = 128
  lr = 0.0001
  gamma = 0.99
  TAU = 0.05
  
  Memory = deque(maxlen=100000)

  Env1 = gym.make("CartPole-v1",render_mode='rgb_array')
  action_size = Env1.action_space.n
  observation_size = Env1.observation_space.shape[0]

  policy_net=DQN(action_size,observation_size).to(device)
  target_net=DQN(action_size,observation_size).to(device)
  target_net.load_state_dict(policy_net.state_dict())
  optimizer=optim.AdamW(policy_net.parameters(),lr=lr)
  loss_fn=nn.MSELoss()

  policy_net.train()
  target_net.eval()

  steps_done  = 0
  sc = []
  for episode in range(episodes):
    state,_ = Env1.reset()
    state = torch.tensor(state)
    done = False
    score = 0
    while not done:

      if eps >0.05:
        eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
      else:
        eps_threshold = 0.05

      if np.random.randn()<eps_threshold:
        action = Env1.action_space.sample()
      else:
        with torch.no_grad():
          action = torch.argmax(policy_net(state.view(1,4).to(device))).item()

      next_state,reward,done,terminated,_ = Env1.step(action)
      next_state = torch.tensor(next_state)
      score += reward
      

      # done = done or terminated
      Memory.append((state,action,reward,next_state,done))
      state = next_state 
    sc.append(score)


    if len(Memory)>Memory_size:

         batch = random.sample(Memory,batch_size)
         batch_state,batch_action,batch_reward,batch_next_state,batch_done = zip(*batch)
         

         batch_state=torch.stack(batch_state).to(device)
         batch_next_state=torch.stack(batch_next_state).to(device)
         batch_state = batch_state.to(device)
         batch_next_state = batch_next_state.to(device)

         batch_action=torch.tensor(batch_action).to(device)
         batch_reward=torch.tensor(batch_reward).view(len(batch_reward),1).to(device)
         batch_done = np.array(batch_done)
         batch_done = batch_done.astype(int)
         batch_done=torch.tensor(batch_done).view(len(batch_done),1).to(device)


         policy_net.zero_grad()
         batch_q = policy_net(batch_state)
         with torch.no_grad():
            batch_next_q = target_net(batch_next_state)
         batch_next_q = torch.max(batch_next_q,dim=1)[0]
         batch_target = batch_reward + gamma * batch_next_q.unsqueeze(1) * (1-batch_done)
         batch_q_action = batch_q.gather(1,batch_action.unsqueeze(1)).squeeze(1)
        

         loss = loss_fn(batch_q_action,batch_target.squeeze(1))
         loss.backward()
         torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
         optimizer.step()


    if (episode%100==0):
      print(f"Episode {episode} Score {score} ")

    if episode%40==0:
      target_net_state_dict = target_net.state_dict()
      policy_net_state_dict = policy_net.state_dict()
      for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
      target_net.load_state_dict(target_net_state_dict)

    if score>300:
      torch.save(policy_net.state_dict(),'model.pth')

  return sc
