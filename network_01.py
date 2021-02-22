import numpy as np

# y= x_1 + 2x_2
input= np.arr([[1, 1], [1, 2], [2, 1], [2, 2]])
ans= np.arr([3, 5, 4, 6])

epoch= 10
weight= np.random.rand(4)
bias= np.random.rand(4)

y_h= []

for _ in range(epoch):
    # forward
    for i in input:
        y_1= np.dot(i, np.transpose(weight))+ bias
        y_h.append(y_1)

    # calc loss
    y_sub= ans - y_h
    rmse= np.sqrt(np.sum(y_sub * y_sub) / len(ans))