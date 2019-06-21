import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    d0 = pd.read_csv('./data/train.csv')
    l = d0['label'].to_numpy()
    d = d0.drop('label', axis=1).to_numpy()
    return d, l
