import os
import glob
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_datasets(datasets, N):
    X, y = [], []
    for i in range(N, len(datasets)):
        X.append(datasets.iloc[i - N:i][['lng', 'lat', 'MMSI']].values)
        y.append(datasets.iloc[i][['lng', 'lat', 'MMSI']].values)
    return torch.Tensor(np.array(X)).float().to(device), torch.Tensor(np.array(y)).float().to(device)


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

        # 定义switch_gate的线性层
        self.switch_gate_i = nn.Linear(input_size + hidden_size, 1)
        self.switch_gate_g = nn.Linear(input_size + hidden_size, 1)

        init.normal_(self.switch_gate_i.weight, mean=0.0, std=0.01)
        init.normal_(self.switch_gate_i.bias, mean=0.0, std=0.01)
        init.normal_(self.switch_gate_g.weight, mean=0.0, std=0.01)
        init.normal_(self.switch_gate_g.bias, mean=0.0, std=0.01)

    def forward(self, input_tensor, cur_state):
        global c_next
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        i = torch.sigmoid(self.input_gate(combined))
        f = torch.sigmoid(self.forget_gate(combined))
        g = torch.tanh(self.cell_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))

        w_i = torch.sigmoid(self.switch_gate_i(combined)).squeeze(1)
        w_g = torch.sigmoid(self.switch_gate_g(combined)).squeeze(1)

        c_next_i = f * c_cur + i
        c_next_g = f * c_cur + g
        c_next_ig = f * c_cur + i * g

        count_w_g_greater = (w_g > p).sum().item()
        count_w_i_greater = (w_i > p).sum().item()

        total_elements = w_g.numel()
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
        self.lstm_layers = nn.ModuleList([
            SimpleLSTMCell(input_size, hidden_size),
            *[SimpleLSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        ])
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, c = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)], \
            [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                input_ = x[:, t, :] if layer == 0 else h[layer - 1]
                h[layer], c[layer] = self.lstm_layers[layer](input_, (h[layer], c[layer]))
        out = self.linear(h[-1])
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(datasets, N, input_size, hidden_size, num_layers, output_size, learning_rate, n_epochs, batch_size):
    start_time = time.time()
    scaler = MinMaxScaler()
    datasets[['lng', 'lat', 'MMSI']] = scaler.fit_transform(datasets[['lng', 'lat', 'MMSI']])

    X, y = create_datasets(datasets, N)
    X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), test_size=0.2,
                                                        random_state=42)
    X_train = torch.Tensor(X_train).float().to(device)
    X_test = torch.Tensor(X_test).float().to(device)
    y_train = torch.Tensor(y_train).float().to(device)
    y_test = torch.Tensor(y_test).float().to(device)
    print(X_train.shape, y_train.shape)
    X_train = X_train.view(X_train.shape[0], N, -1)
    X_test = X_test.view(X_test.shape[0], N, -1)

    model = LSTM_Predictor(input_size, hidden_size, num_layers, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)

    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        train_losses.append(epoch_loss)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_train_pred = model(X_train)
                train_rmse = np.sqrt(loss_fn(y_train_pred.cpu(), y_train.cpu()).item())
                y_test_pred = model(X_test)
                test_rmse = np.sqrt(loss_fn(y_test_pred.cpu(), y_test.cpu()).item())
                test_losses.append(test_rmse)
            print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, Test RMSE {test_rmse:.4f}")

            # 保存模型
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} trainable parameters.")

    torch.save(model.state_dict(), 'SI-LSTM_model1.pth')
    print(f"Model saved to 'SI-LSTM_model1.pth'")
    # 记录结束时间并计算总训练时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    return scaler, train_losses, test_losses
def preprocess_data(dataset):
    dataset[['lng', 'lat', 'MMSI']] = scaler.transform(dataset[['lng', 'lat', 'MMSI']])
    X, y = create_datasets(dataset, N)
    X = X.view(X.shape[0], N, -1)
    X = X.to(device)
    y = y.to(device)
    return X, y
def predict_and_evaluate(dataset):
    model_path = 'SI-LSTM_model.pth'
    model = LSTM_Predictor(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    X, y = preprocess_data(dataset)
    with torch.no_grad():
        y_pred = model(X)

    y_pred_original = scaler.inverse_transform(y_pred.cpu().numpy())
    y_original = scaler.inverse_transform(y.cpu().numpy())

    mae_lat = mean_absolute_error(y_original[:, 0], y_pred_original[:, 0])
    mae_lng = mean_absolute_error(y_original[:, 1], y_pred_original[:, 1])
    rmse_lat = np.sqrt(mean_squared_error(y_original[:, 0], y_pred_original[:, 0]))
    rmse_lng = np.sqrt(mean_squared_error(y_original[:, 1], y_pred_original[:, 1]))

    average_mae = (mae_lat + mae_lng) / 2
    average_rmse = (rmse_lat + rmse_lng) / 2

    return average_mae, average_rmse, y_pred_original

input_size = 3
hidden_size = 64
num_layers = 3
output_size = 3
learning_rate = 0.001
p = 0.5
a = 0.5
N=3
n_epochs = 100
loss_fn = nn.MSELoss()

datasets = pd.read_csv('Case 1.csv', encoding='ISO-8859-1')
scaler, train_losses, test_losses = train_model(datasets, N, input_size, hidden_size, num_layers, output_size, learning_rate, n_epochs, 8)
datasets[['lng', 'lat', 'MMSI']] = scaler.fit_transform(datasets[['lng', 'lat', 'MMSI']])
case_files = glob.glob('Case *.csv')
case_results = []
predicted_coordinates_all = []

for idx, case_file in enumerate(sorted(case_files)):
    case_name = os.path.basename(case_file)
    print(f"Processing {case_name}...")

    dataset = pd.read_csv(case_file, encoding='utf-8-sig')
    mae, rmse, y_pred_original = predict_and_evaluate(dataset)

    pred_df = pd.DataFrame(y_pred_original[:, :2], columns=['lng', 'lat'])
    pred_df['case'] = case_name
    file_name = f'SI-LSTM_case_{idx + 1}.xlsx'
    pred_df.to_excel(file_name, index=False)

    case_results.append((case_name, mae, rmse))
    print(f"{case_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

print("\nAll Case Results:")
for case_name, mae, rmse in case_results:
    print(f"{case_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
