import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

#################
# Read Datasets #
#################
train = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

train, validation = torch.utils.data.random_split(train, [0.8, 0.2])

######################
# Create Dataloaders #
######################
batch_size = 64

train_load = DataLoader(train, batch_size=batch_size, shuffle=True)
val_load = DataLoader(validation, batch_size=batch_size, shuffle=True)
test_load = DataLoader(test, batch_size=batch_size, shuffle=True)


#########################
# Create the Classifier #
#########################
class GarmentClassifier(nn.Module):
    """
    I took the main code from here:
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-model

    then added a bunch of Linear layers to get the parameters up to 500.000
    """
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 500)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("total number of parameters: ", pytorch_total_params)

#################
# Training Loop #
#################
num_epochs = 10
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, num_epochs + 1):
    train_correct = 0
    val_correct = 0

    for i, data in enumerate(train_load):
        input, label = data

        # training
        model.train()
        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        train_correct += (torch.argmax(output, dim=1) == label).sum()

    train_score = train_correct / len(train_load)

    # validation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_load):
            val_input, val_label = data

            val_output = model(val_input)
            val_loss = criterion(val_output, val_label)

            val_correct += (torch.argmax(val_output, dim=1) == val_label).sum()

    val_score = val_correct / len(val_load)

    # print every epoch
    print(f"epoch #{epoch}")
    print(f"train-acc {train_score: .2f}%, loss {loss: .6f}, val-acc: {val_score: .2f}%, val-loss: {val_loss: .6f}")
