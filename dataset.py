import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_dataset():
    # import datasets
    train = pd.read_csv('input/train.csv').set_index('id')

    # Label Encoder for Vehicle_Age
    le = LabelEncoder()

    # train
    train['Gender'] = train['Gender'].apply(lambda i: 1 if i == 'Female' else 0)
    train['Vehicle_Damage'] = train['Vehicle_Damage'].apply(lambda i: 1 if i == 'Yes' else 0)
    train['Vehicle_Age'] = le.fit_transform(train['Vehicle_Age'])

    # train, validation and test dataset preparation
    y = train['Response']
    X = train.drop(columns=['Response'])

    # Splitting to train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    # Standardization with StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val
