import numpy as np
from utils.plots import *

class Visualize:
    def __init__(self, data, target, title, labels = ['X', 'Y']):
        self.target = target
        self.data = data
        self.title = title
        self.labels = labels
    
    def visualize(self, type='2D'):
        try:
            x_cuts, y_cuts = self._convert_to_cuts()
        except:
            x_cuts, y_cuts = [], []

        if type == '2D':
            plot2D(self.data, self.target, x_cuts, y_cuts, self.title, self.labels[0], self.labels[1])

    def _convert_to_cuts(self) -> tuple[list[float], list[float]]:
        # Extracts the cuts from data:  
        # ["0.9; 1.95", "5.35; 6.8", 3],
        # ["1.95; 4.2", "5.35; 6.8", 2],
        # ["1.95; 4.2", "2.1; 5.35", 1],
        # ["0.9; 1.95", "2.1; 5.35", 2],
        # ["1.95; 4.2", "5.35; 6.8", 2],
        # ["1.95; 4.2", "2.1; 5.35", 1],
        # ["0.9; 1.95", "5.35; 6.8", 3]
        # like
        # dict = {
        #     'x1': [0.9, 1.95, 4.2],
        #     'x2': [2.1, 5.35, 6.8]
        # }
        # except for dict['x*'][0] and dict['x*'][-1] are not cuts, they are points instead.
        # so even something like this will do:
        # dict = {
        #     'first_x': 0.9,
        #     'first_y': 2.1,
        #     'last_x': 4.2,
        #     'last_y': 4.8,
        #     'x1': [1.95],
        #     'x2': [5.35]
        # }
        pass