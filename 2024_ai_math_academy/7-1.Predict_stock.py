"""
주식 가격 예측하기
"""

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
# 아무튼 제일 앞 주식종목코드만 변경하면 특정 종목에서 데이터 가져오기 가능!

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


# ===== Model ===== #

model = Sequential()
model.add(Dense(1000, input_dim=5, activation='relu')) # 1000 neurons
model.add(Dense(100, activation='relu')) # 100 neurons with tanh activation function
model.add(Dense(500, activation='relu')) # 500 neurons
model.add(Dense(1, activation='sigmoid')) # 1 output neuron

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=500, batch_size=10,  verbose=0, validation_data=(X_val, y_val))

"""
# ===== Visualization ===== #

plt.plot(history.history['loss'], ".-", label='loss')
plt.plot(history.history['accuracy'], ".-")
plt.plot(history.history['val_loss'], ".-", label='val_loss')
plt.plot(history.history['val_accuracy'], ".-")
plt.legend()
plt.show()
"""

# ===== Evaluation ===== #
scores = model.evaluate(scaler.transform(X_test), y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(scaler.transform(X_test))
rounded = [int(np.round(x, 0)) for x in predictions]
print(rounded)


# ===== Lv3. Setup Envorinment ===== #

clf = setup(data=data.drop('Diff', axis=1), target='Label')


# ===== Lv4. Find Model ===== #

top5_models = compare_models(fold=5, round=3, sort='AUC', n_select=5)
# print(top5_models) # ridge , lda , lr , qda , et


# ===== Lv5. Create Models ===== #

model_ridge = create_model('ridge', fold=5)
model_lda = create_model('lda', fold=5)
model_lr = create_model('lr', fold=5)
model_qda = create_model('qda', fold=5)
model_et = create_model('et', fold=5)

total_models = [model_ridge, model_lda, model_lr, model_qda, model_et]


# ===== Lv6. Tuning Model ===== #

model_ridge = tune_model(model_ridge, fold=5, optimize = 'AUC', choose_better = True)
model_lda = tune_model(model_lda, fold=5, optimize = 'AUC', choose_better = True)
model_lr = tune_model(model_lr, fold=5, optimize = 'AUC', choose_better = True)
model_qda = tune_model(model_qda, fold=5, optimize = 'AUC', choose_better = True)
model_et = tune_model(model_et, fold=5, optimize = 'AUC', choose_better = True)

tuned_models = [model_ridge, model_lda, model_lr, model_qda, model_et]

"""
튜닝한 모델보다 기존(original model)의 성능이 더 좋음을 알 수 있었다.
"""

# ===== Lv7. Plot Models ===== #

# plot_model(model_lr, plot='auc') # AUC Plot
# plot_model(model_lr, plot='pr') # Precision-Recall Curve
# plot_model(model_lr, plot='feature') # Feature Importance Plot
# plot_model(model_lr, plot = 'confusion_matrix') # Confusion Matrix
# plt.show()


# ===== Lv8. Predict for validation data ===== #

# for model in tuned_models:
#     display(predict_model(model))


# ===== Lv9. Finalize for Deployment ===== #

final_model = finalize_model(model_ridge)


# ===== Lv10. Prediction ===== #

prediction = predict_model(final_model, data=X_test)
print("============================")
print("This is the prediction model")
print("============================")
print(prediction)


# ===== Lv11. Save Model ===== #

save_model(final_model, 'stock_predict_20240820')


# ===== Lv12. Load Model ===== #

# saved_final_model = load_model('stock_predict_20240820')
# new_prediction = predict_model(saved_final_model, data=X_test)
# print(new_prediction.head())