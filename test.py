import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load and preprocess data
x = "humidity"
y = "temp"
z = "winddir"
s = "sealevelpressure"

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Check if 'precip' column exists and create 'rain' column accordingly
if 'precip' in train_df.columns:
    train_df["rain"] = (train_df["precip"] > 0).astype(int)
else:
    train_df["rain"] = 0  # Default value if 'precip' column is missing

if 'precip' in test_df.columns:
    test_df["rain"] = (test_df["precip"] > 0).astype(int)
else:
    test_df["rain"] = 0  # Default value if 'precip' column is missing

train_df = train_df[[x, y, z, s, "rain"]]
test_df = test_df[[x, y, z, s, "rain"]]

# Standardize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_df[[x, y, z, s]])
test_features = scaler.transform(test_df[[x, y, z, s]])

train_labels = train_df["rain"].values
test_labels = test_df["rain"].values

# Convert to PyTorch tensors with gradients enabled
train_features = torch.tensor(train_features, dtype=torch.float32, requires_grad=True)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Check if GPU is available and move tensors and model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_features = train_features.to(device)
train_labels = train_labels.to(device)
test_features = test_features.to(device)
test_labels = test_labels.to(device)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 5)  # Reduce the number of units
        self.fc2 = nn.Linear(5, 2)
        self.drp = nn.Dropout(0.1)  # Increase the dropout rate
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.drp(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Initialize variables for early stopping
best_loss = float('inf')
patience = 10
trigger_times = 0

# Train the model on GPU with progress bar
num_epochs = 20  # Reduce the number of epochs

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(range(1), desc=f"Training Epoch {epoch+1}/{num_epochs}")

    for _ in progress_bar:
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == train_labels).sum().item()
        total += train_labels.size(0)
        
        accuracy = 100 * correct / total
        progress_bar.set_postfix(loss=total_loss, accuracy=accuracy)
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Early stopping
    if total_loss < best_loss:
        best_loss = total_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

# Evaluate the model on GPU
model.eval()
with torch.no_grad():
    outputs = model(test_features)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == test_labels).sum().item() / len(test_labels) * 100
    print(f'Final Accuracy: {accuracy:.2f}%')
