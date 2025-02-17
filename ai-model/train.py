# file for training the models

# importing the required modules
import torch
import torch.nn as nn
import torch.optim  as optim
import torchvision.transforms as transforms
import torchvisin.datasets as datasets
import torch.utils.data as Dataloader

# setting the up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using  device {device}")

# dataset directory
data_dir = "dataset"

# define image transformation 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# loading the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = Dataloader(dataset, batchsize=32, shuffle=True)

# get class names
class_names = dataset.classes
print(f"classes found {class_names}")

# define cnn models