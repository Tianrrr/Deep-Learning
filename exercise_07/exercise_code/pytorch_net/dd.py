import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
 
# 1.准备数据集
batch_size = 64
# batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # 将输入的图片转化成张量
    transforms.Normalize((0.1307,), (0.3081,))  # 对输入的图片进行归一化
])
train_dataset = datasets.MNIST(root='../mydatasets/mnist/',
                               train=True,  # 作为训练集
                               download=True,  # 如果没有下载就自动下载
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='../mydatasets/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)
 
 
# 2.设计模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
 
    def forward(self, x):
        x = x.view(-1, 784)  # 将输入的张量转化为1列784行
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 直接输出第五层的数据，后面直接进入CrossEntropyLoss()
 
 
model = Net()
 
# 3.损失和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 学习率0.01，冲量0.5
 
 
# 4.训练和测试
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data  # inputs是输入x，target是真实值y
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:  # 将每次训练后得到一个loss，300个loss取平均值使曲线平滑
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
 
 
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 无需计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 输出的数据是一个N×10的矩阵，N表示图片数量为N，10行代表10各分类的概率，取出概率最大值（dim=1）
            total += labels.size(0)  # 计算第一列的数量（测试集样本总数）
            correct += (predicted == labels).sum().item()  # 如果预测的结果等于真实值标签，那么就把这个数记录到correct里面
    print('Accuracy on test set: %d %%' % (100 * correct / total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
