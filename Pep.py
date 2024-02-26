import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim


class Peppy(nn.Module):
    def __init__(self):
        super(Peppy, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(24*24*64, 512)
        self.fc2 = nn.Linear(512, 10)  # Adjust to output 10 reward predictions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 24*24*64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit the model
        transforms.ToTensor(),  # Convert game states to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
    ])

    dataset = ImageFolder(root='States', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the nagging back seat driver
    discriminator = Peppy()
    discriminator.to('cuda:0')

    # Optimizer
    optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 1000  # Or however many epochs you deem necessary
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
    
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = discriminator(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f'Loss: {loss.item()}')

    # After training, you can save the model if you want
    torch.save(discriminator.state_dict(), 'Peppy.pth')
