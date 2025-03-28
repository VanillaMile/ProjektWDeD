from matplotlib import pyplot as plt
import numpy as np

def plot2D(data, target, x_cuts, y_cuts, title, x_label, y_label):
    if data.shape[1] != 2:
        raise ValueError("Data must have 2 features for 2D visualization.")
    
    fig = plt.figure(figsize=(5, 5))

    axes = fig.add_subplot(111)
    scatter = axes.scatter(data[:, 0], data[:, 1], c=target, cmap='viridis', s=50, alpha=0.8)

    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    for cut in x_cuts:
        plt.axvline(cut, color='r')
    for cut in y_cuts:
        plt.axhline(cut, color='b')

    levels = np.arange(target.min(), target.max() + 2)
    plt.colorbar(scatter, ticks=levels[:-1], label='Classes')

    plt.show()