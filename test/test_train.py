from data_loader import get_loader
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net
import csv
import os

def train_net(net, device, train_loader, epochs=350, lr=0.00001):
    f = open('train.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(["Epoch", "Total_Epoch", "loss"])
    f.close()
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train(True)
        # 按照batch_size开始训练
        print('epoch:', epoch)
        average_loss = 0
        for i, (image, label) in enumerate(train_loader):
        # for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # SR_probs = F.sigmoid(pred)
            # print(SR_probs)
            # print(label)
            # SR_flat = SR_probs.view(SR_probs.size(0), -1)
            # 计算loss
            loss = criterion(pred, label)
            # print('loss:', loss)
            average_loss += loss.item()
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        average_loss = average_loss / (i + 1)
        f = open('train.csv', 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([epoch + 1, epochs, average_loss])
        f.close()
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道3，分类为1。
    net = U_Net(img_ch=3, output_ch=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    net = nn.DataParallel(net)
    # 指定训练集地址，开始训练
    # data_path = "data/train/"
    train_loader = get_loader(image_path="../data_set/Urben_pre/train/",
                              image_size=256,
                              batch_size=8,
                              num_workers=8,
                              mode='train',
                              augmentation_prob=0)
    train_net(net, device, train_loader)