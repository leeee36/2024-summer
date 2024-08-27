import numpy
import matplotlib.pyplot as plt

X = list(range(1, 10))
Y = list(range(11, 20))

# plt.plot(X, Y)
# plt.scatter(X, Y)
# plt.show()

# ----------------------------------

class H():
    def __init__(self, w):
        self.w = w

    def forward(self, x):
        return self.w * x
    
def cost(h, X, Y):
    error = 0

    for i in range(len(X)):
        error += (h.forward(X[i]) - Y[i]) ** 2
    error = error / len(X)
    return error

list_w = []
list_c = []

for i in range(10):
    w = i * 0.1
    h = H(w)
    c = cost(h, X, Y)
    
    list_w.append(w)
    list_c.append(c)

# # plt.figure(figsize=(1,1)) # 그래프 사이즈 지정
# plt.title("first graph")
# plt.xlabel("w")
# plt.ylabel("cost")
# plt.scatter(list_w, list_c)
# plt.show()

def grad(w, cost):
    h = H(w)
    cost1 = cost(h, X, Y)
    eps = 10e-3

    h = H(w+eps)
    cost2 = cost(h, X, Y)
    
    dcost = cost2 - cost1
    dw = eps

    grad = dcost / dw
    return grad, (cost1+cost2)*0.5

def grad2(w, cost):
    h = H(w)
    grad = 0
    for i in range(len(X)):
        grad += (h.forward(X[i]) - Y[i]) * X[i]
    grad = grad / len(X)
    c = cost(h, X, Y)
    return grad, c

w1, w2 = 5, 5 
lr = 10e-4
for i in range(10):
    gradient1, mean_cost1 = grad(w, cost)
    gradient2, mean_cost2 = grad(w, cost)
    
    w1 -= lr * gradient1
    w2 -= lr * gradient2

    print(w1, mean_cost1)
    print(w2, mean_cost2)