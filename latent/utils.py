import matplotlib.pyplot as plt
import numpy as np

def draw_samples_2d(data, labels):
    num_labels = len(np.unique(labels))

    plt.figure(figsize=(16, 9))
    for i in range(num_labels):
        plt.scatter(data[labels == i, 0], data[labels == i, 1])
