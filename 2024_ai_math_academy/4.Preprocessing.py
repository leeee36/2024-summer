import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import matplotlib.pyplot as plt


X = np.array([[2104, 3],
              [1600, 3],
              [2400, 3],
              [1416, 2],
              [3000, 4],
              [1985, 4],
              [1534, 3],
              [1427, 3],
              [1380, 3],
              [1494, 3]]) # [면적, 화장실 개수]
y = np.array([[1, 0],
              [1, 0],
              [1, 0],
              [0, 1],
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0],
              [1, 0],
              [1, 0]]) # [good, bad]
newX = preprocessing.scale(X)

model = Sequential()
model.add(Dense(units=5, activation='relu', input_dim=2))
model.add(Dense(units=5, activation='relu', input_dim=5))
model.add(Dense(units=2, activation='softmax', input_dim=5))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(newX, y, epochs=1000, verbose=1)
model.summary()

pred = np.round(model.predict(newX), 2)
print(pred)

plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()