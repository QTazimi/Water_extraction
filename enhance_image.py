import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

#进行样本的数据增强，负样本数量太少的时候可以对负样本进行数据增强。
def get_enhance_image(path, rotation_degree, reduction_ratio):
    #旋转放缩数据但会造成一定的黑边。
    #path原始图片的路径，rotation_degree旋转角度，reduction_ratio放缩比率
    img = cv2.imread(path)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 旋转中心坐标，逆时针旋转：45°，缩放因子：0.5
    M_1 = cv2.getRotationMatrix2D(center, rotation_degree, reduction_ratio)
    rotated_1 = cv2.warpAffine(img, M_1, (w, h))

    plt.imshow(rotated_1)
    plt.show()

    # cv2.imshow('rotated_45.jpg', rotated_1)
    temp_path = path.split('.')[0]
    rear = path.split('.')[1]
    new_path = temp_path + '_' + str(rotation_degree) + '_' + str(reduction_ratio).replace('.', 'd') + '.' + rear
    print(new_path)
    # cv2.imwrite(new_path, rotated_1)

# get_enhance_image('D:/Workspace/Pycharm/Unet/data_set/Urben_original/train\images_png/1379.png', 120, 0.8)

def get_rotation_image(path, ro):
    #只旋转3种旋转角度:90、180、270
    img = Image.open(path)
    temp_path = path.split('.')[0]
    rear = path.split('.')[1]
    # new_path = temp_path + '_' + ro + '.' + rear
    # print(new_path)
    if ro == '90':
        new_img = img.transpose(Image.ROTATE_90)  # 将图片旋转90度
        plt.imshow(new_img)
        plt.show()
        # new_img.show("img/rotateImg.png")
        # new_img.save(new_path)
    elif ro == '180':
        new_img = img.transpose(Image.ROTATE_180)  # 将图片旋转180度
        # new_img.show("img/rotateImg.png")
        # new_img.save(new_path)
    elif ro == '270':
        new_img = img.transpose(Image.ROTATE_270)  # 将图片旋转270度
        # new_img.show("img/rotateImg.png")
        # new_img.save(new_path)

get_rotation_image('D:/Workspace/Pycharm/Unet/data_set/Urben_original/train\images_png/1379.png', "90")
def get_all_enhance(path):
    lists = os.listdir(path)
    for item in lists:
        temp_path = path + item
        print(temp_path)
        get_rotation_image(temp_path, '180')

# get_all_enhance("D:/HAAR/data_set/negative_enhance_dataset/")


# path = "D:/HAAR/test/1/106.jpg"
# get_rotation_image(path, "90")
# get_enhance_image(path, 45, 0.5)