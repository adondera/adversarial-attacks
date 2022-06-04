import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AdversarialPatchLog:
    def __init__(self):
        self.train_scores = []
        self.test_scores = []
        self.fig, self.ax = plt.subplots()

        plt.ion()
        plt.draw()

    def plot(self):
        epochs = np.arange(0, len(self.train_scores))
        self.ax.clear()
        self.ax.plot(epochs, self.train_scores, label='Test success rate', linewidth=2, color='r')
        self.ax.plot(epochs, self.test_scores, label='Train success rate', linewidth=2, color='b')
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Success rate")
        self.ax.legend()

        plt.draw()
        plt.pause(0.02)

        self.fig.savefig("training_pictures/patch_attack_success_rate.png")

    def save_log(self):
        df = pd.DataFrame(data={
            "Epochs": np.arange(0, len(self.train_scores)),
            "Train score": self.train_scores,
            "Test score": self.test_scores
        })
        df.to_csv("log.csv")
