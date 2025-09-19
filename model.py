import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# Define simple CNN
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)  # Good or Bad

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_export():
    # -----------------------------
    # Data transforms
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("datasets/train", transform=transform)
    val_data = datasets.ImageFolder("datasets/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    # -----------------------------
    # Training setup
    # -----------------------------
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # Train
    # -----------------------------
    for epoch in range(5):  # adjust as needed
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # -----------------------------
    # Save model
    # -----------------------------
    torch.save(model.state_dict(), "cnn_model.pth")
    print("✅ Training complete. Model saved as cnn_model.pth")

    # -----------------------------
    # Export to ONNX
    # -----------------------------
    dummy_input = torch.randn(1, 3, 128, 128)
    torch.onnx.export(model, dummy_input, "saved_model.onnx",
                      input_names=["input"], output_names=["output"])
    print("✅ Model exported to saved_model.onnx")


if __name__ == "__main__":
    train_and_export()
