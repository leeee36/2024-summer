import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)

x_new = np.array([[20]])
y_pred = model.predict(x_new)
print(y_pred)

# ===== visualization ===== #
fig = plt.figure()
fig.suptitle('Linear Regression')

ax1 = fig.add_subplot(2,2,1)
ax1.scatter(X, y) # 점 그래프
ax1.set_xlabel('X')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(X, y) # 선 그래프
ax2.set_xlabel('X')
ax2.set_ylabel('y')

plt.show()