import pandas as pd
import torch
import numpy as np
# Models

import torch.nn as nn
import vec

# Training

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# Evaluation
import random

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 42 + 512
HIDDEN_SIZE = 256
OUT_DIM = 6
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 50


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # self.bert = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2") #不再引入bert模型 因为向量化的工作已经完成
        # self.embedding =   # 这里用向量化函数 vector
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(2 * HIDDEN_SIZE, OUT_DIM)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text, hidden):
        # tk = random.randint(1, 5)
        # if tk == 3:
        #     print(text)
        text_emb = text.view(len(text), 1, -1)
        # text_emb = torch.tensor(text_emb)         # new
        out_put, hidden = self.lstm(text_emb, hidden)
        text_out = out_put[:, -1, :]
        text_out = self.fc(text_out)
        text_out = self.softmax(text_out)
        return text_out, hidden

    def init_hidden(self, batch_size=BATCH_SIZE):
        h0 = torch.zeros((1 * 2, batch_size, HIDDEN_SIZE)).detach().to(DEVICE)
        c0 = torch.zeros((1 * 2, batch_size, HIDDEN_SIZE)).detach().to(DEVICE)
        hidden = (h0, c0)
        return hidden




model = LSTM()
model = model.to(DEVICE)
criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-6)
# embedding 测试
# text_emb = torch.tensor(vec.vector("今天天气不错，大概过几天楼下的花就要开了"))
# print(text_emb)


# usual_train
# 0 neutral 5749
# 1 angry 8344
# 2 happy 5379
# 3 surprise 2086
# 4 sad 4990
# 5 fear 1220

df = pd.read_csv("usual_train.csv")
df1 = pd.read_csv("usual_eval_labeled.csv")
# print(df.head())
x = df['text']
X = []
cnt = 0
for items in x:
    # print(items)
    to_vec = vec.vector(items)
    X.append(to_vec)
    cnt += 1
    if cnt % 5000 == 0:
        print(cnt)
# print(np.shape(X))
y = df['label']
X = np.array(X)
y = np.array(y)
x1 = df1['text']
X1 = []
y1 = df1['label']
y1 = np.array(y1)
for items in x1:
    to_vec = vec.vector(items)
    X1.append(to_vec)
X1 = np.array(X1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=219)
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
test_data = TensorDataset(torch.from_numpy(X1), torch.from_numpy(y1))
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
total_step = len(train_loader)
total_step_val = len(valid_loader)

early_stopping_patience = 5000
early_stopping_counter = 0

valid_acc_max = 0  # Initialize best accuracy top 0

for e in range(EPOCHS):

    # lists to host the train and validation losses of every batch for each epoch
    train_loss, valid_loss = [], []
    # lists to host the train and validation accuracy of every batch for each epoch
    train_acc, valid_acc = [], []

    # lists to host the train and validation predictions of every batch for each epoch
    y_train_list, y_val_list = [], []

    # initalize number of total and correctly classified texts during training and validation
    correct, correct_val = 0, 0
    total, total_val = 0, 0
    running_loss, running_loss_val = 0, 0

    ####TRAINING LOOP####

    model.train()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # load features and targets in device
        # print(inputs)
        # print(labels)
        h = model.init_hidden(labels.size(0))

        model.zero_grad()  # reset gradients

        output, h = model(inputs, h)  # get output and hidden states from LSTM network

        loss = criterion(output, labels)
        loss.backward()

        running_loss += loss.item()

        optimizer.step()

        y_pred_train = torch.argmax(output, dim=1)  # get tensor of predicted values on the training set
        y_train_list.extend(y_pred_train.squeeze().tolist())  # transform tensor to list and the values to the list
        # print(y_pred_train)
        # print(labels)
        correct += torch.sum(y_pred_train == labels).item()  # count correctly classified texts per batch
        total += labels.size(0)  # count total texts per batch

    train_loss.append(running_loss / total_step)
    train_acc.append(100 * correct / total)

    ####VALIDATION LOOP####

    with torch.no_grad():

        model.eval()

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            val_h = model.init_hidden(labels.size(0))

            output, val_h = model(inputs, val_h)

            val_loss = criterion(output, labels)
            running_loss_val += val_loss.item()

            y_pred_val = torch.argmax(output, dim=1)
            y_val_list.extend(y_pred_val.squeeze().tolist())

            correct_val += torch.sum(y_pred_val == labels).item()
            total_val += labels.size(0)

        valid_loss.append(running_loss_val / total_step_val)
        valid_acc.append(100 * correct_val / total_val)

    # Save model if validation accuracy increases
    if np.mean(valid_acc) >= valid_acc_max:
        torch.save(model.state_dict(), './state_dict.pt')
        print(
            f'Epoch {e + 1}:Validation accuracy increased ({valid_acc_max:.6f} --> {np.mean(valid_acc):.6f}).  Saving model ...')
        valid_acc_max = np.mean(valid_acc)
        early_stopping_counter = 0  # reset counter if validation accuracy increases
    else:
        print(f'Epoch {e + 1}:Validation accuracy did not increase')
        early_stopping_counter += 1  # increase counter if validation accuracy does not increase

    if early_stopping_counter > early_stopping_patience:
        print('Early stopped at epoch :', e + 1)
        print(f'\tTrain_loss : {np.mean(train_loss):.4f} Val_loss : {np.mean(valid_loss):.4f}')
        print(f'\tTrain_acc : {np.mean(train_acc):.3f}% Val_acc : {np.mean(valid_acc):.3f}%')
        break

    print(f'\tTrain_loss : {np.mean(train_loss):.4f} Val_loss : {np.mean(valid_loss):.4f}')
    print(f'\tTrain_acc : {np.mean(train_acc):.3f}% Val_acc : {np.mean(valid_acc):.3f}%')

#evel
model.load_state_dict(torch.load('./state_dict.pt'))
model.eval()
y_pred_list = []
y_test_list = []
for inputs, labels in test_loader:
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    test_h = model.init_hidden(labels.size(0))
    output, val_h = model(inputs, test_h)
    y_pred_test = torch.argmax(output, dim=1)
    y_pred_list.extend(y_pred_test.squeeze().tolist())
    y_test_list.extend(labels.squeeze().tolist())
lenl = len(y_pred_test)
cntl = 0
for i in range(lenl):
    if y_pred_test[i] == y_test_list[i]:
        cntl += 1

print("Accuracy is ", float(cntl/lenl))