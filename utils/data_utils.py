from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_data(filepath):
    data = pd.read_csv(filepath).values
    return data

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, labels=None):
    if labels is not None:
        return TensorDataset(torch.FloatTensor(data), torch.FloatTensor(labels))
    return TensorDataset(torch.FloatTensor(data))

def split_dataset(data, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(data, test_size=test_size, random_state=random_state)

def save_results(results, filepath):
    np.savetxt(filepath, results, delimiter=',')