import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from temporal_cross_validation import temporal_random_split_by_horizon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# load train and test dataset
train_samples,test_samples,vali_samples = temporal_random_split_by_horizon(train_size=0.8, test_size=0.1, vali_size=0.1,random_seed=42)

X_train = train_samples.drop(columns=["UpDown_next"]).values
y_train = train_samples["UpDown_next"].values
X_vali = vali_samples.drop(columns=["UpDown_next"]).values
y_vali = vali_samples["UpDown_next"].values
X_test = test_samples.drop(columns=["UpDown_next"]).values
y_test = test_samples["UpDown_next"].values

# 转成 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_vali = torch.tensor(X_vali, dtype=torch.float32)
y_vali = torch.tensor(y_vali, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
vali_dataset = TensorDataset(X_vali, y_vali)
train_loader = DataLoader(train_dataset, batch_size=300, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)
vali_loader = DataLoader(vali_dataset, batch_size=30, shuffle=False)


class StockNN(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=32, output_dim=2):
        super(StockNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)  
        self.fc2 = nn.Linear(hidden1, hidden2)    
        self.fc3 = nn.Linear(hidden2, output_dim) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # x = self.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x


model = StockNN(input_dim=X_train.shape[1])

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)

# using train and vali data totrain model and check performance 
epochs = 300
train_losses = []
val_losses = []
val_accs = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    #
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in vali_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)
    
    avg_val_loss = val_loss / len(vali_loader)
    val_acc = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.4f}")

# plot loss figure
plt.figure(figsize=(8, 5))
plt.plot(range(11, epochs + 1), train_losses[10:], marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('training_loss2.png', dpi=200)

# acc test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
