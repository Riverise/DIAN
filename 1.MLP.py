import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import utils

random_seed = 42
torch.manual_seed(random_seed)

# Load MNIST data

X_train, y_train = utils.load_mnist(r'E:\CODE\JupyterWork\DIAN2024春招算法\mnist', kind='train')
X_test, y_test = utils.load_mnist(r'E:\CODE\JupyterWork\DIAN2024春招算法\mnist', kind='t10k')


class Net(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size):
        # auto-adjust the network according to the params
        super(Net, self).__init__()
        layers = [nn.Linear(input_size, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_size))
        self.mlp = nn.Sequential(*layers)
        self.config = str(hidden_dims)

    def forward(self, x):
        return self.mlp(x)


sns.set(style='whitegrid')
plt.figure(figsize=(10, 5))
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')


def train(num_epochs, batch_size, net, optimizer, criterion, plot=False):
    net.train()
    losses = []
    for epoch in range(num_epochs):
        for batch_idx, (batch_X, batch_y) in enumerate(utils.data_iter(batch_size, X_train, y_train)):
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # print(f'Epoch {epoch + 1}-batch {batch_idx},loss:{loss.item()}')
    # Visualization
    if plot:
        plt.plot(losses, label=net.config)
        plt.legend(loc='best')


def evaluate(net, criterion):
    net.eval()
    with torch.no_grad():
        outputs = net(X_test)
        test_loss = criterion(outputs, y_test)
        _, predicted = torch.max(outputs, 1)
        f1 = utils.get_macro_f1(predicted, y_test)
        accuracy = utils.get_accuracy(predicted, y_test)
    return test_loss.item(), f1, accuracy


configs = [
    {'input_size': 784, 'hidden_dims': [128], 'output_size': 10},
    {'input_size': 784, 'hidden_dims': [128, 128], 'output_size': 10},
    {'input_size': 784, 'hidden_dims': [128, 128, 128], 'output_size': 10},
    {'input_size': 784, 'hidden_dims': [128, 256], 'output_size': 10},
    {'input_size': 784, 'hidden_dims': [256, 256], 'output_size': 10},
    {'input_size': 784, 'hidden_dims': [256, 384], 'output_size': 10},
    {'input_size': 784, 'hidden_dims': [384, 384], 'output_size': 10},
]

config_infos = []
test_losses = []
f1_scores = []
accuracies = []

epoch = 3
lr = 0.005
bs = 1000

for config in configs:
    net = Net(**config)
    criterion = nn.CrossEntropyLoss()
    sgd = optim.SGD(net.parameters(), lr=lr)
    train(num_epochs=epoch, batch_size=bs, net=net, optimizer=sgd, criterion=criterion, plot=True)
    test_loss, f1, accuracy = evaluate(net, criterion=criterion)

    # store relevant data
    config_infos.append(str(config['hidden_dims']))
    test_losses.append(test_loss)
    f1_scores.append(f1)
    accuracies.append(accuracy)

df = pd.DataFrame({'test_loss': test_losses, 'f1_score': f1_scores, 'accuracy': accuracies}, index=config_infos)
print(df)
# explore the potential effect of train_steps

configure = {'input_size': 784, 'hidden_dims': [128, 256], 'output_size': 10}

bs_list = [500, 1000, 1500, 2000, 3000]
test_losses = []
f1_scores = []
accuracies = []

for bs in bs_list:
    basic = Net(**configure)  # reset parameter
    criterion = nn.CrossEntropyLoss()
    sgd = optim.SGD(basic.parameters(), lr=lr)
    train(num_epochs=epoch, batch_size=bs, net=basic, optimizer=sgd, criterion=criterion)
    test_loss, f1, accuracy = evaluate(basic, criterion=criterion)

    test_losses.append(test_loss)
    f1_scores.append(f1)
    accuracies.append(accuracy)

df2 = pd.DataFrame({'test_loss': test_losses, 'f1_score': f1_scores, 'accuracy': accuracies}, index=bs_list)
print("Explore the effect of train steps based on the configure [128, 256]")
print(df2)
plt.show()

"""
在其余条件不变的情况下：
1.神经网络层数越多，拟合能力越强，最终在测试集上的拟合效果越好。
2.隐藏层神经元个数（即神经网络的宽度）并不是越多越好，也并不是越少越好；
3.训练步数（取决于batch_size）越多，模型的泛化效果更好。
"""