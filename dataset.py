import random
import os
from PIL import Image

def split_train_test(data, train_ratio1, valid_ratio2):
    # 设置随机数种子，保证每次生成的结果都是一样的
    rng = random.Random(12345)
    rng.shuffle(data)
    train_set_size = int(len(data) * train_ratio1)
    valid_set_size = int(len(data) * valid_ratio2)
    # test_set_size = len(data) - train_set_size - valid_set_size
    train_datas = data[:train_set_size]
    valid_datas = data[train_set_size:train_set_size + valid_set_size]
    test_datas = data[train_set_size + valid_set_size:]
    return train_datas, valid_datas, test_datas


def get_dataset(original_path, save_path):
    lists = os.listdir(original_path)
    print(len(lists))
    train_datas, valid_datas, test_datas = split_train_test(lists, 0.7, 0.2)
    print(len(train_datas), len(valid_datas), len(test_datas))
    for item in train_datas:
        train_path = original_path + item
        save_train_path = save_path + 'train/' + item
        image = Image.open(train_path)
        image.save(save_train_path)
        # image.show()
    for item in valid_datas:
        valid_path = original_path + item
        save_valid_path = save_path + 'valid/' + item
        image = Image.open(valid_path)
        image.save(save_valid_path)
    for item in valid_datas:
        test_path = original_path + item
        save_test_path = save_path + 'test/' + item
        image = Image.open(test_path)
        image.save(save_test_path)

    print("end!")




# get_dataset("E:/高分辨率/many/", "E:/高分辨率/new_splict/")