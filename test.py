from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from kan import KAN, KANLinear
from torchmetrics.classification import MulticlassAccuracy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale()]
)
training_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
validation_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

class KAN_MNIST_model(nn.Module):
    
    def __init__(self, input_size=32*32) -> None:
        super(KAN_MNIST_model, self).__init__()
        self.input_size = input_size
        
        self.model = torch.nn.Sequential(
            KAN([32*32, 128, 64, 10])
        ).to('cuda')
        
    def forward(self, X):
        X = X.view(-1, self.input_size)
        return self.model(X)
    
class MLP_MNIST_model(nn.Module):
    
    def __init__(self, input_size=32*32) -> None:
        super(MLP_MNIST_model, self).__init__()
        self.input_size = input_size
        
        self.model = torch.nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )
        
    def forward(self, X):
        X = X.view(-1, self.input_size)
        return self.model(X)

train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(validation_set, batch_size=64, shuffle=False)

def train(model, dataloader, loss_fn, optimizer, scheduler):
    # Train Model
    model.train()
    
    total_loss = 0.
    for i, (features, labels) in enumerate(dataloader):
        features = features.to('cuda')
        labels = labels.to('cuda')
        
        output = model(features.float())
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"\t{i+1} Current Loss: {total_loss / 100}")
            total_loss = 0.
    scheduler.step()
    
    # Validate on training_set
    model.eval()
    
    mca = MulticlassAccuracy(num_classes=10).to('cuda')
    total_output = None
    total_labels = None
    with torch.inference_mode():
        for i, (features, labels) in enumerate(dataloader):
            features = features.to('cuda')
            labels = labels.to('cuda')
            
            output = model(features.float())
            if total_output is None:
                total_output = output
            else:
                torch.cat((total_output, output))
            
            if total_labels is None:
                total_labels = labels
            else:
                torch.cat((total_labels, labels))
        print(f"\tTraining Accuracy: {mca(total_output.to('cuda'), total_labels.to('cuda')) * 100:.2f}%")
    return mca(total_output.to('cuda'), total_labels.to('cuda')).item()


def validate(model, dataloader):
    # Validate on testing_set
    model.eval()
    
    mca = MulticlassAccuracy(num_classes=10).to('cuda')
    total_output = None
    total_labels = None
    with torch.inference_mode():
        for i, (features, labels) in enumerate(dataloader):
            features = features.to('cuda')
            labels = labels.to('cuda')
            
            output = model(features.float())
            if total_output is None:
                total_output = output
            else:
                torch.cat((total_output, output))
            
            if total_labels is None:
                total_labels = labels
            else:
                torch.cat((total_labels, labels))
                
    print(f"\tValidation Accuracy: {mca(total_output.to('cuda'), total_labels.to('cuda')) * 100:.2f}%")
    return mca(total_output.to('cuda'), total_labels.to('cuda')).item()
            
model = KAN_MNIST_model().to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)
num_epochs = 10

kan_training_accuracy = []
kan_testing_accuracy = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}:")
    kan_training_accuracy.append(train(model, train_dataloader, loss_fn, optimizer, scheduler))
    kan_testing_accuracy.append(validate(model, test_dataloader))
    
    
model = MLP_MNIST_model().to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

mlp_training_accuracy = []
mlp_testing_accuracy = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}:")
    mlp_training_accuracy.append(train(model, train_dataloader, loss_fn, optimizer, scheduler))
    mlp_testing_accuracy.append(validate(model, test_dataloader))
    
x = range(10)
plt.plot(x, kan_training_accuracy, label='KAN Training')
plt.plot(x, kan_testing_accuracy, label="KAN Testing")
plt.plot(x, mlp_training_accuracy, label="MLP Training")
plt.plot(x, mlp_testing_accuracy, label='MLP Testing')
plt.legend()
plt.title("Model Performance Comparison")
plt.show()