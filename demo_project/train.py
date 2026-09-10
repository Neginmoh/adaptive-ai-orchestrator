import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Create a small synthetic classification dataset.
x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

dataset = TensorDataset(x, y)

# Create the training data loader.
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

# Define the model and training components.
model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


# Train the model.
for epoch in range(5):
    for features, labels in train_loader:
        optimizer.zero_grad()

        predictions = model(features)
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()