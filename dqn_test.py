import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym

device ='cuda' if torch.cuda.is_available() else 'cpu'

def test(DQN,episodes):

  Env1 = gym.make("CartPole-v1",render_mode='human')
  action_size = Env1.action_space.n
  observation_size = Env1.observation_space.shape[0]

  policy_net=DQN(action_size,observation_size).to(device)
  policy_net.load_state_dict(torch.load('model.pth'))

  sc = []
  for episode in range(episodes):
    state,_ = Env1.reset()
    state = torch.tensor(state)
    done = False
    score = 0
    while not done:

      with torch.no_grad():
          action = torch.argmax(policy_net(state.view(1,4).to(device))).item()

      next_state,reward,done,terminated,_ = Env1.step(action)
      next_state = torch.tensor(next_state)
      score += reward
      state = next_state 
    print(f" episode {episode} score {score}")
    sc.append(score)
  Env1.close()
  return sc
