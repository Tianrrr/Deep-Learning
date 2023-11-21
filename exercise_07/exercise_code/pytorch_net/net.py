import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision


classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


batch_size = 64


transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, )), transforms.RandomHorizontalFlip()])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])



train_dataset = torchvision.datasets.FashionMNIST(root="../mydatasets/FashionMNIST", train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="../mydatasets/FashionMNIST", train=False, transform=transform_test, download=True)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = torch.nn.Linear(784, 512)
        self.layer_2 = torch.nn.Linear(512, 256)
        self.layer_3 = torch.nn.Linear(256, 128)
        self.layer_4 = torch.nn.Linear(128, 64)
        self.layer_5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        _, _, H, D = x.size()
        x = x.view(-1, H * D)  
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        return self.layer_5(x)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        m.bias.data.fill_(0.01)


model = Net()
model.apply(init_weights)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimzier = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # inputs, target = inputs.to(device), target.to(device)

        optimzier.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimzier.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/2000))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            # inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)
            total+= target.size(0)
            correct += preds.eq(target).sum().item()
    print('Accuracy on validation set: %d %% [%d/%d]' % (100 * correct / total, correct, total))

def show_img():
    dataiter = iter(test_loader)
    images, labels = dataiter.__next__()
    # get sample outputs
    outputs = model(images)
    # convert output probabilites to predicted class
    _, predicted = torch.max(outputs, 1)

    fig = plt.figure(figsize=(25,4))  # 宽25 长4 英寸
    for idx in range(8):
        ax = fig.add_subplot(2, 8//2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')

        ax.set_title(f"{classes[predicted[idx]]} ({classes[labels[idx]]})",
                    color="green" if predicted[idx]==labels[idx] else "red")
    plt.show()


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
    show_img()

















