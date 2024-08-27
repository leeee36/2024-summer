import pandas as pd
import numpy as np
from pycaret.classification import *
from time import time

import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense


# ===== Load Dataset ===== #

company_df = pd.read_excel("상장법인목록.xlsx", engine='openpyxl')
df = pdr.DataReader("005930", "naver", start="2024-01-01", end="2024-08-16")
# datareader("불러올 주식 종목코드", "naver finance에서 정보를 가져오겠다", ...)

df = df.astype(float)
df['Diff'] = df['Close'] - df['Open']
df['Diff'] = df['Close'].diff()
df['Label'] = df['Diff'].apply(lambda x: 1 if x > 0 else (0 if x < 0 else None))
# print(df[['Diff', 'Label']])

data = df.dropna()
X = data.drop("Diff", axis=1).drop("Label", axis=1)
Y = data['Label']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X = scaler.fit_transform(X_train_full) # 전처리 된 데이터
    
X_train, X_val, y_train, y_val = train_test_split(X, y_train_full, test_size=0.25, random_state=1)


# ===== Load Model ===== #
saved_final_model = load_model('stock_predict_20240820')
new_prediction = predict_model(saved_final_model, data=X_test)
print(new_prediction.head())