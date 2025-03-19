import torch
import matplotlib.pyplot as plt
import random


def create_data(w, b, data_num):  # 生成数据
    x = torch.normal(0, 1, (data_num, len(w)))
    y = torch.matmul(x, w) + b

    noise = torch.normal(0, 0.01, y.shape)
    y += noise

    return x, y


num = 500

true_w = torch.tensor([10.0, 7.0, 5.0, 2.0])
true_b = torch.tensor(1.1)

X, Y = create_data(true_w, true_b, num)


def data_provider(data, label, batchsize):
    length = len(label)
    indices = list(range(length))
    random.shuffle(indices)

    for each in range(0, length, batchsize):
        get_indices = indices[each: each + batchsize]
        get_data = data[get_indices]
        get_label = label[get_indices]

        yield get_data, get_label


batchsize = 16


def fun(x, w, b):
    pred_y = torch.matmul(x, w) + b
    return pred_y


def maeLoss(pred_y, y):
    return torch.sum(abs(pred_y-y))/len(y)


def sgd(paras, lr):
    with torch.no_grad():
        for para in paras:
            para -= para.grad * lr
            para.grad.zero_()


lr = 0.05

w_0 = torch.normal(0, 0.01, true_w.shape, requires_grad=True)
b_0 = torch.tensor(0.01, requires_grad=True)
print(w_0, b_0)

epochs = 50

for epoch in range(epochs):
    data_loss = 0
    for batch_x, batch_y in data_provider(X, Y, batchsize):
        pred_y = fun(batch_x, w_0, b_0)
        loss = maeLoss(pred_y, batch_y)
        loss.backward()
        sgd([w_0, b_0], lr)
        data_loss += loss

    print("epoch %03d: loss: %.6f"%(epoch, data_loss))

print("真实的函数值是", true_w, true_b)
print("深度学习得到的函数值是", w_0, b_0)

idx = 0
plt.plot(X[:, idx].detach().numpy(), X[:, idx].detach().numpy()*w_0[idx].detach().numpy()+b_0.detach().numpy())
plt.scatter(X[:, idx], Y, 1)
plt.show()