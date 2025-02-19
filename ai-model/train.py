# file for training the models

# importing the required modules
import torch
import torch.nn as nn
import torch.optim  as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as DataLoader
import os

# setting the up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using  device {device}")

# dataset directory
data_dir = "./ai-model/dataset"

# define image transformation 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# loading the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader.DataLoader(dataset, batch_size=32, shuffle=True)

# get class names
class_names = dataset.classes
print(f"classes found {class_names}")

# define cnn models
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return x

# initialise the model
num_classes = len(class_names)
model = CNN(num_classes).to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss:{running_loss/len(dataloader):.4f}")

# Ensure the model directory exists
model_dir = "./ai-model/model"
os.makedirs(model_dir, exist_ok=True)

# save the trained models
torch.save(model.state_dict(), os.path.join(model_dir, "image_model.pth"))
print("✅ Model training complete and saved!")