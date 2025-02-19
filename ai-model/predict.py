# for predicting the uploaded image

# importing the required modules
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys

# load trained model
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 32 * 32 * 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x

# defining the class names
class_names = ["cat","dog"]

# load model and set to evaluation mode
num_classes = len(class_names)
model = CNN(num_classes)
model.load_state_dict(torch.load("model/image_image.pth", map_location=torch.device("cpu")))
model.eval()

# defining image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# function to predict the image class
def predict_image(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0) # add batch dimension

    with torch.no_grad():
        output = model(img)
        class_idx = torch.argmax(output, dim=1).item()

    return class_names[class_idx]


# getting image path from cli
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_image(image_path)
    print("result", result)