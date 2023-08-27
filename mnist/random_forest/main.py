"""
    Starting file to process mnist dataset 
    creates a decision tree using DecisionTreeClassifier
    with ensemble learning method random forest classifier.
"""
from sklearn.datasets import load_digits
#from sklearn.tree import DecisionTreeClassifier

mnist = load_digits()
