from data_loader import get_loader
from evaluation import *
from network import U_Net
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import csv


def test_net(test_loader, save_path=""):
    net = U_Net(img_ch=3, output_ch=1)
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 将网络拷贝到deivce中
    net = nn.DataParallel(net)
    net.to(device=device)
    # 加载模型参数
    # net.load_state_dict(torch.load("/home/program/Unet/models/U_Net-350-0.0002-70-0.0000.pth"))
    net.load_state_dict(torch.load("best_model.pth", map_location=device))
    # 测试模式
    net = net.module.to(device)
    net.eval()
    print(len(train_loader))
    for i, (image_path, image) in enumerate(test_loader):
        # 将数据拷贝到device中
        image = image.to(device=device, dtype=torch.float32)
        # # 预测
        pred = F.sigmoid(net(image))
        # print(pred)
        # 处理结果
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0


        image_path = image_path[0]
        # 找轮廓
        pred = np.array(pred, np.uint8)
        contours,_ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.imread(image_path, 1)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        img = img[:, :, ::-1]
        img[..., 2] = np.where(pred == 255, 200, img[..., 2])

        plt.imshow(img)
        plt.show()

        filename = image_path.split('/')[-1][:-len(".png")]
        cv2.imwrite(save_path + filename + ".png", pred)


if __name__ == "__main__":
    train_loader = get_loader(image_path="../data_set/test_image/test/",
                              image_size=256,
                              batch_size=1,
                              num_workers=8,
                              mode='test',
                              augmentation_prob=0)#单张图片作为一个batch_size
    test_net(train_loader, save_path="../data_set/test_image/res/")