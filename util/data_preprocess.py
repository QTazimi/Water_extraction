from PIL import Image
import numpy as np
import os


def find_river_image(masks_path, images_path, masks_save_path, images_save_path):
    #Look up remote sensing images containing water areas and modify multi-category labels.
    lists = os.listdir(masks_path)
    for item in lists:
        original_path = masks_path + item
        original_image = Image.open(original_path)
        original_img = np.array(original_image)
        if 4 in original_img:
            print(item)
            original_img[original_img != 4] = 0
            original_img[original_img == 4] = 255
            save_path1 = masks_save_path + item
            img1 = Image.fromarray(original_img)
            img1.save(save_path1)

            original_path2 = images_path + item
            save_path2 = images_save_path + item
            original_image2 = Image.open(original_path2)
            original_img2 = np.array(original_image2)
            img2 = Image.fromarray(original_img2)
            img2.save(save_path2)

    return True

def split_train_image(masks_path, images_path, masks_save_path, images_save_path):
    #1024*1024 remote sensing images were cut into 16 256*256 images
    lists = os.listdir(masks_path)
    print(len(lists))
    count = 0#0
    for item in lists:
        temp_path1 = masks_path + item
        im1 = Image.open(temp_path1)
        temp_path2 = images_path + item
        im2 = Image.open(temp_path2)
        # 准备将图片切割成16张小图片
        size = im1.size
        weight = int(size[0] // 4)
        height = int(size[1] // 4)
        for j in range(4):
            for i in range(4):
                box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
                region1 = im1.crop(box)
                region2 = im2.crop(box)
                img_numpy = np.array(region1)
                if 255 in img_numpy:
                    print(count)
                    path1 = masks_save_path + str(count) + ".png"
                    path2 = images_save_path + str(count) + ".png"
                    count += 1
                    region1.save(path1)
                    region2.save(path2)
    return True

def split_test_image(images_path, images_save_path):
    # 1024*1024 remote sensing images were cut into 16 256*256 images
    lists = os.listdir(images_path)
    print(len(lists))
    for item in lists:
        temp_path = images_path + item
        im = Image.open(temp_path)
        # 准备将图片切割成16张小图片
        size = im.size
        weight = int(size[0] // 4)
        height = int(size[1] // 4)
        for j in range(4):
            for i in range(4):
                box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
                region = im.crop(box)
                # img_numpy = np.array(region)
                path = images_save_path + item + "_" + str(j) + str(i) + ".png"
                region.save(path)
    return True

def combination_test_image(images_path, images_save_path):
    pass
# find_river_image("/home/program/Unet/data_set/Urben_original/val/masks_png/",
#                  "/home/program/Unet/data_set/Urben_original/val/images_png/",
#                  "/home/program/Unet/data_set/Urben_original/valid/masks_png/",
#                  "/home/program/Unet/data_set/Urben_original/valid/images_png/")

# split_train_image("/home/program/Unet/data_set/Urben_original/valid/masks_png/",
#                   "/home/program/Unet/data_set/Urben_original/valid/images_png/",
#                   "/home/program/Unet/data_set/Urben_pre/annotation/",
#                   "/home/program/Unet/data_set/Urben_pre/train/")

# lists = os.listdir("/home/program/Unet/data_set/Urben_pre/train/")
# print(len(lists))



