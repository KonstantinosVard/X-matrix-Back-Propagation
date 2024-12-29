import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class KernelFitter(nn.Module):
    def __init__(self, kernel_size=3, num_kernels=5):
        super(KernelFitter, self).__init__()
        # Define a learnable kernel layer
        self.conv_layer = nn.Conv2d(1, num_kernels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(num_kernels * 14 * 14, 10)  # Assuming kernel size fits 28x28 input

    def forward(self, x):
        x = self.conv_layer(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# Loading the KMNIST dataset
def load_kmnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training function
def train_model(model, train_loader, epochs=5, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 250 == 0:  # Print every 250 batches
                print(f"Epoch {epoch}, Batch {i}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0
                
                
                
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

def classify_sample(model, sample):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(sample.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    return predicted.item()

def visualize_prediction(model, sample, true_label):
    """
    Visualizes the input sample, its true label, and the model's prediction.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(sample.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)

    # Convert the tensor to a NumPy array for visualization
    image = sample.squeeze(0).numpy()

    # Plot the image with true and predicted labels
    plt.imshow(image, cmap="gray")
    plt.title(f"True Label: {true_label}, Predicted: {predicted.item()}")
    plt.axis("off")
    plt.show()