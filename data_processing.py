# Data processing
import pandas as pd
import numpy as np


class DataProcessing:
    def __init__(self, data):
        """
            Class for data processing
        """
        self.data = data
        self.data_points = pd.DataFrame(self.data['data'])
        self.labels = pd.DataFrame(self.data['target'])

    def get_data_points(self):
        """
        Returns data points
        Returns:
            _type_: DataFrame
        """
        return self.data_points

    def get_specific_data_point(self, data_set, index=0):
        """
        Return specific data point
        Args:
            data_set (_type_): DataFrame
            index (int, optional): Index for which to data point to pick. Defaults to 0.

        Returns:
            _type_: DataFrame data point
        """
        return data_set.loc[index]

    def get_labels(self):
        """
        Returns labels
        Returns:
            _type_: DataFrame
        """
        return self.labels

    def get_specific_label(self, data_set, index=0):
        """_summary_

        Args:
            data_set (_type_): DataFrame
            index (_type_): Index for which to data point to pick. Defaults to 0.

        Returns:
            _type_: DataFrame target point
        """
        return data_set.loc[index]

    def get_data(self):
        """
        Returns dataset
        Returns:
            _type_: Dataset
        """
        return self.data
