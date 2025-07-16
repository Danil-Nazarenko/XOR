import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(42)
W1 = np.random.uniform(-1, 1, (2, 2))  
b1 = np.zeros((1, 2))                

W2 = np.random.uniform(-1, 1, (2, 1)) 
b2 = np.zeros((1, 1))                  

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a1, a2

def backprop(X, y, a1, a2, lr=0.1):
    global W1, W2, b1, b2

    error = y - a2
    d_a2 = error * sigmoid_derivative(a2)

    dW2 = np.dot(a1.T, d_a2)
    db2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)

    dW1 = np.dot(X.T, d_a1)
    db1 = np.sum(d_a1, axis=0, keepdims=True)

    W2 += lr * dW2
    b2 += lr * db2
    W1 += lr * dW1
    b1 += lr * db1

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    a1, a2 = forward(X)
    backprop(X, y, a1, a2, learning_rate)

    if epoch % 1000 == 0:
        loss = np.mean((y - a2) ** 2)
        print(f"Эпоха {epoch}, ошибка: {loss:.4f}")

print("\nИтоговое предсказание (округлено):")
print(np.round(a2))

print("\nФактические значения предсказаний (без округления):")
print(a2)
