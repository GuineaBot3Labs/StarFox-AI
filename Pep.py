import torch
import threading
import os
import shutil
import schedule
import time
from torch import nn, optim
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import torchvision.transforms.functional as TF

def classify_existing_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            print(f"Classifying existing file: {filename}")
            classify_new_image(file_path)

def classify_new_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)  # Assuming 'transform' is defined in the global scope
        image = image.unsqueeze(0).to(device)

        peppy.eval()  # Ensure Peppy is in evaluation mode
        with torch.no_grad():
            output = peppy(image)
            predicted = output.argmax(1)
            print(f"Classified as: {predicted.item()}")

        # Move the file to its new classified folder
        target_folder = os.path.join(folder, str(predicted.item()))
        os.makedirs(target_folder, exist_ok=True)
        shutil.move(image_path, os.path.join(target_folder, os.path.basename(image_path)))
        print(f"Moved {os.path.basename(image_path)} to {target_folder}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


class Peppy(nn.Module):
    def __init__(self):
        super(Peppy, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Dynamically calculate the flattened size after convolutions
        self._to_linear = None
        
        # Dummy pass to get the size that will be fed into the first linear layer
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU()
        )
        self._get_conv_output_size([3, 224, 224])

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 10)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            self._to_linear = torch.numel(self.convs(torch.zeros(1, *shape)))
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten the output for the FC layer
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def update_peppy(update_epochs):
    print("Updating Peppy with new data...")
    # Assuming 'dataloader' is globally defined and updated with any new data
    global peppy, dataloader
    peppy.train()  # Ensure Peppy is in training mode
    for epoch in range(update_epochs):  # 'update_epochs' is the number of epochs for each update cycle
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = peppy(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Update Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    torch.save(peppy.state_dict(), peppy_model_path)  # Save the updated model
    print("Peppy updated and ready to classify new data.")

    # Schedule the next update
    threading.Timer(20 * 60, update_peppy).start()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    folder = input("Enter your folder name: ")
    choice = input("Does Peppy need training? (y|n): ")
    dataset = datasets.ImageFolder(root=folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu" # Get Debugging info
    peppy = Peppy().to(device)

    # Check if the Peppy model exists
    peppy_model_path = 'Peppy.pth'
    if os.path.isfile(peppy_model_path):
        peppy.load_state_dict(torch.load(peppy_model_path, map_location=device))
        print("Loaded Peppy model from", peppy_model_path)
    else:
        print("No saved Peppy model found at", peppy_model_path)

    optimizer = optim.Adam(peppy.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    if choice == "y":
        # Training loop
        num_epochs = 10  # Adjust epochs to your needs
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = peppy(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        torch.save(peppy.state_dict(), peppy_model_path)
    else:
        print("Skipping for now.")
        
    path_to_watch = "Unlabeled"  # Adjust this path to your unlabeled data directory

    while True:
        update_epochs = 4
        classify_existing_images(path_to_watch)
        print("Waiting 4 minutes for more data...")
        time.sleep(4 * 60)
        classify_existing_images(path_to_watch)
        update_peppy(update_epochs)
        print("Waiting 4 minutes for more data...")
        
