# Tree Visualization
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from data_processing import DataProcessing


class ModelVisualization:
    def __init__(self):
        pass

    def plot_image(self, content, label):
        """
        Plots image
        Args:
            content (_type_): data
            label (_type_): target
        """
        point = np.array(content, dtype='uint8')
        point = point.reshape((8, 8))

        plt.title('Label is {label}'.format(label=label))
        plt.imshow(point, cmap='gray')
        plt.show()

    def plot_heatmap(self, cm):
        """
        Plot heatmap using confusion matrix
        """
        plt.figure(figsize=(5, 5))
        sn.heatmap(cm, annot=True)
        plt.xlabel('predicted')
        plt.ylabel('truth')
        plt.show()
