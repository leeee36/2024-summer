import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# ===== dataset ===== #

dataset = pd.read_csv('pima-indians-diabetes.csv',
                      names=['임신횟수', '포도당농도', '혈압', '피부주름두께', '인슐린', '체질량', '혈통', '나이', '결과'])

X = dataset.drop('결과', axis=1)
y = dataset['결과']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123)


"""
# ===== visualization data ===== #

# dataset['결과'].plot.hist()

result_one = np.sum(dataset['결과'] == 1)
result_zero = np.sum(dataset['결과'] == 0)
plt.pie([result_one, result_zero],
        labels=['1', '0'],
        autopct='%.2f',
        shadow=True
        )

plt.show()
"""


# ===== model ===== #

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=500)
# model.evaluate(X_test, y_test)


# ===== visualization ===== #
fig = plt.figure(figsize=(10,5))
fig.suptitle('Loss & Accuracy')

ax1 = fig.add_subplot(1,2,1)
ax1.set_title('loss')
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='test')
ax1.legend()

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('accuracy')
ax2.plot(history.history['accuracy'], label='train')
ax2.plot(history.history['val_accuracy'], label='test')
ax2.legend()

plt.show()