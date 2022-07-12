from data_loader import get_loader
from evaluation import *
from network import U_Net
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import csv

def get_one_image(image_path):
    image = Image.open(image_path)
    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = Transform(image)
    Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = Norm_(image)
    return [image_path, image]


def test_one_image(test_loader, save_path=""):
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

    ##########################################
    image_path, image = test_loader[0], test_loader[1]
    image = torch.unsqueeze(image, dim=0)
    # 将数据拷贝到device中
    image = image.to(device=device, dtype=torch.float32)
    # # 预测
    pred = F.sigmoid(net(image))
    # print(pred)
    # 处理结果
    pred = np.array(pred.data.cpu()[0])[0]
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0


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
    save_path = save_path + filename + "_res.png"
    # cv2.imwrite(save_path, pred)
    img = img[:, :, ::-1]
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    test_loader = get_one_image("1049.png")

    path = test_one_image(test_loader, save_path="../data_set/test_image/res/")#返回值是具体的存储路径../data_set/test_image/res/1049.png