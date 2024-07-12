import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasets = pd.read_csv('smoothed_trajectory.csv', encoding='ISO-8859-1')
N = 3
def create_datasets(datasets, N):
    X, y = [], []
    for i in range(N, len(datasets)):
        X.append(datasets.iloc[i - N:i][['lng', 'lat']].values)
        y.append(datasets.iloc[i][['lng', 'lat']].values)
    return torch.Tensor(np.array(X)).float().to(device), torch.Tensor(np.array(y)).float().to(device)

scaler = MinMaxScaler()
datasets[['lng', 'lat']] = scaler.fit_transform(datasets[['lng', 'lat']])

X, y = create_datasets(datasets, N)

X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), test_size=0.2, random_state=42)
X_train = torch.Tensor(X_train).float().to(device)
X_test = torch.Tensor(X_test).float().to(device)
y_train = torch.Tensor(y_train).float().to(device)
y_test = torch.Tensor(y_test).float().to(device)
print(X_train.shape)

X_train = X_train.view(X_train.shape[0], N, -1)
X_test = X_test.view(X_test.shape[0], N, -1)
print(X_train.shape)

class SimpleLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义四个线性层（输入门、遗忘门、单元门、输出门）
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)


    def forward(self, input_tensor, cur_state):
        global c_next
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        i = torch.sigmoid(self.input_gate(combined))
        f = torch.sigmoid(self.forget_gate(combined))
        g = torch.tanh(self.cell_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))


        c_next_i = f * c_cur + i
        c_next_g = f * c_cur + g
        c_next_ig = f * c_cur + i * g

        # 计算满足条件的元素数量
        count_w_g_greater = (g > p).sum().item()
        count_w_i_greater = (i > p).sum().item()

        # 获取张量元素的总数
        total_elements = g.numel()  # 假设 w_g 和 w_i 的形状相同，元素数量也相同

        # 检查是否有一半以上的元素满足条件
        if count_w_g_greater < total_elements * a and count_w_i_greater > total_elements * a:
            c_next = c_next_i
        elif count_w_g_greater > total_elements * a and count_w_i_greater < total_elements * a:
            c_next = c_next_g
        else:c_next = c_next_ig
        h_next = o * torch.tanh(c_next)


        return h_next, c_next


class LSTM_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 初始化LSTM层
        self.lstm_layers = nn.ModuleList([
            SimpleLSTMCell(input_size, hidden_size),
            *[SimpleLSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        ])

        # 初始化输出层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h, c = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)], \
            [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        # 按时间步迭代
        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                # 当前层的输入
                input_ = x[:, t, :] if layer == 0 else h[layer - 1]
                # LSTM单元前向传播
                h[layer], c[layer] = self.lstm_layers[layer](input_, (h[layer], c[layer]))

                # 取最后一个时间步的输出
        out = self.linear(h[-1])
        return out


input_size = 2
hidden_size = 20
num_layers = 2
output_size = 2
p=0.5
a=0.5
learning_rate = 0.001

model = LSTM_Predictor(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=10)


n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train.view(X_train.shape[0], N, -1))
            train_rmse = np.sqrt(loss_fn(y_train_pred.cpu(), y_train.cpu()).item())
            y_test_pred = model(X_test.view(X_test.shape[0], N, -1))
            test_rmse = np.sqrt(loss_fn(y_test_pred.cpu(), y_test.cpu()).item())
        print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, Test RMSE {test_rmse:.4f}")

model.eval()
with torch.no_grad():
    y_pred_all = model(X).cpu().numpy()
    y_true_all = y.cpu().numpy()
y_pred_all_original = scaler.inverse_transform(y_pred_all)
y_true_all_original = scaler.inverse_transform(y_true_all)


