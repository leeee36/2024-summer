from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# ===== data ===== #

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# AND
y1 = np.array([[0],
              [0],
              [0],
              [1]])
# OR
y2 = np.array([[0],
              [1],
              [1],
              [1]])
# XOR
y3 = np.array([[0],
              [1],
              [1],
              [0]])


# ===== class ===== #

# class net(Sequential):
    
#     def __init__(self):
#         super(net, self).__init__()
        
#     def _make_layers(self):
#         self.add(Dense(units=8, activation='tanh', input_dim=2))
#         self.add(Dense(units=1, activation='sigmoid'))
#         self.compile(loss='mean_squared_error', optimizer='adam')
        

# ===== model ===== #

model1 = Sequential()
model1.add(Dense(units=8, activation='tanh', input_dim=2))   # tanh
model1.add(Dense(units=1, activation='sigmoid'))             # sigmoid
model1.compile(loss='mean_squared_error', optimizer='adam')  # adam
history1 = model1.fit(X, y1, epochs=1000, verbose=1)

model2 = Sequential()
model2.add(Dense(units=8, activation='tanh', input_dim=2))   # tanh
model2.add(Dense(units=1, activation='sigmoid'))             # sigmoid
model2.compile(loss='mean_squared_error', optimizer='adam')  # adam
history2 = model2.fit(X, y2, epochs=1000, verbose=1)

model3 = Sequential()
model3.add(Dense(units=8, activation='tanh', input_dim=2))   # tanh
model3.add(Dense(units=1, activation='sigmoid'))             # sigmoid
model3.compile(loss='mean_squared_error', optimizer='adam')  # adam
history3 = model3.fit(X, y3, epochs=1000, verbose=1)


# ===== result ===== #
pred1 = np.round(model1.predict(X), 2)
pred2 = np.round(model2.predict(X), 2)
pred3 = np.round(model3.predict(X), 2)
print(f'Result of AND : {pred1}')
print(f'Result of OR  : {pred2}')
print(f'Result of XOR : {pred3}')


# ===== visualization ===== #
fig = plt.figure()
fig.suptitle('Loss of each model')

# AND
ax1 = fig.add_subplot(2,3,1)
ax1.plot(history1.history['loss'])
ax1.set_title('AND')
ax1.set_xlabel('X')
ax1.set_ylabel('y')

# OR
ax2 = fig.add_subplot(2,3,2)
ax2.plot(history2.history['loss'])
ax2.set_title('OR')
ax2.set_xlabel('X')
ax2.set_ylabel('y')

# XOR
ax3 = fig.add_subplot(2,3,3)
ax3.plot(history3.history['loss'])
ax3.set_title('XOR')
ax3.set_xlabel('X')
ax3.set_ylabel('y')

plt.show()