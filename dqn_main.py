from dqn_model import DQN
from dqn_train import train
import matplotlib.pyplot as plt
from dqn_test import test



def plot(reward,episodes):
  plt.plot(range(episodes),reward)
  plt.show()


condition = input('ENTER TEST OR TRAIN ')
if condition == 'TRUE':
    reward = train(DQN,10000)
    plot(reward,10000)
else:
  reward = test(DQN,10)
  plot(reward,10)









