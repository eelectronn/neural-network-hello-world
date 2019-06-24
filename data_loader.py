import numpy as np
import pandas as pd


def load_training_data():
    file = pd.read_csv('./data/train.csv')
    label = file['label'].to_numpy()
    data = file.drop('label', axis=1).to_numpy()
    training_data = [data[0:30000], get_expected_vector(label[0:30000])]
    testing_data = [data[30000:], label[30000:]]
    return training_data, testing_data


def get_expected_vector(label):
    vectors = np.zeros((len(label), 10))
    for i in range(len(label)):
        vectors[i][label[i]] = 1.0
    return vectors
