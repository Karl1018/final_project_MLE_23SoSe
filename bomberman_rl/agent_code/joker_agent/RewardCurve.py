import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class RewardCurve():
    def __init__(self):
        self.rewards = []

    def record(self, reward: int):
        self.rewards.append(reward)

    def draw(self, round: int):
        #TODO: recording steps
        plt.clf()
        self.rewards = self.rewards[max(0, len(self.rewards) - 100): ]
        plt.plot(self.rewards)
        plt.xlabel('step')
        plt.ylabel('reward')
        plt.title('Reward Curve')
        plt.savefig(f"logs/reward curves/reward_curve.png")