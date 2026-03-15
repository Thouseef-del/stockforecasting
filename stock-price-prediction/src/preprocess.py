import numpy as np

def preprocess_data(df):

    df['Prediction'] = df['Close'].shift(-1)

    df = df.dropna()

    X = df[['Close']].values

    y = df['Prediction'].values

    return X,y