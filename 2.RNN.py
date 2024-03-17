import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from tqdm import tqdm

random_seed = 42
torch.manual_seed(random_seed)

# Load MNIST data
X_train, y_train = utils.load_mnist(r'E:\CODE\JupyterWork\DIAN2024春招算法\fashion-mnist', kind='train')
X_test, y_test = utils.load_mnist(r'E:\CODE\JupyterWork\DIAN2024春招算法\fashion-mnist', kind='t10k')

# normalization
X_train = X_train / 255.0
X_test = X_test / 255.0


class RNN(nn.Module):
    def __init__(self, seq_length=28, embed_dim=28, hidden_dim=56, output_dim=10):
        # self.seq_length = seq_length
        # self.embed_dim = embed_dim
        # self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        super(RNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        # 初始化权重
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.orthogonal_(self.linear3.weight)

    def forward(self, seq):
        h = torch.zeros(seq.shape[0], self.hidden_dim)
        # (batch_size * seq_length * embed_dim)
        seq = seq.view(seq.shape[0], self.seq_length, -1)
        for i in range(self.seq_length):
            x = seq[:, i, :]
            h = F.tanh(self.linear1(x) + self.linear2(h))
        # Output
        output = self.linear3(h)
        return output


seq_length = 28
embed_dim = 28
hidden_dim = 256
output_dim = 10
learning_rate = 0.1

rnn = RNN(seq_length, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
sgd = optim.SGD(rnn.parameters(), lr=learning_rate)

batch_size = 1000
num_epochs = 15
clip_value = 2.0


def train(num_epochs, batch_size, net, optimizer, criterion, clip_value):
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (batch_X, batch_y) in enumerate(utils.data_iter(batch_size, X_train, y_train)):
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            # losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), clip_value)  # gradient clipping
            optimizer.step()
            print(f"Epoch {epoch + 1}-batch {batch_idx + 1}: loss:{loss.item()}")
            train_losses.append(loss.item())

    # Visualization
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss ')
    plt.xlabel('Batch')
    plt.ylabel('Loss')


train(num_epochs, batch_size, rnn, sgd, criterion, clip_value)

# Evaluation
rnn.eval()
with torch.no_grad():
    outputs = rnn(X_test)
    _, predicted = torch.max(outputs, 1)

    accuracy = utils.get_accuracy(predicted, y_test)
    precision = utils.get_macro_precision(predicted, y_test)
    recall = utils.get_macro_recall(predicted, y_test)
    f1_score = utils.get_macro_f1(predicted, y_test)

    print(f'Accuracy on test set: {accuracy * 100}%')
    print(f'Precision on test set: {precision * 100}%')
    print(f'Recall on test set: {recall * 100}%')
    print(f'F1 Score on test set: {f1_score}')

plt.show()
