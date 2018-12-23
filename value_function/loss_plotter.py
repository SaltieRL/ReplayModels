import matplotlib.pyplot as plt
import numpy as np


class LossPlotter:

    def __init__(self):
        self.fig = plt.figure()
        self.ax = plt.gca()
        self.shown = False

    def update_plot(self, x: np.ndarray, y: np.ndarray):
        self.ax.clear()
        self.ax.scatter(x, y, marker='.', alpha=0.4)
        self.ax.grid()
        self.ax.set_xlabel('Actual values')
        self.ax.set_ylabel('Predicted values')
        self.fig.tight_layout()
        self.fig.canvas.draw()
        if not self.shown:
            plt.show(block=False)
            self.shown = True
        plt.pause(0.001)
