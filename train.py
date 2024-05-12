# Efficient-CapsNet Model Train

# In this notebook we provide a simple interface to train Efficient-CapsNet on the three dataset discussed in "Efficient-CapsNet: Capsule Network with Self-Attention Routing":

# - MNIST (MNIST)
# - smallNORB (SMALLNORB)
# - Multi-MNIST (MULTIMNIST)

# The hyperparameters have been only slightly investigated. So, there's a lot of room for improvements. Good luck!

# **NB**: remember to modify the "config.json" file with the appropriate parameters.

import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages, plotHistory
from models import EfficientCapsNet
import parser
# from utils.transforms_nocrop import build_transforms
import time
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser(description='Training network')
    # 输入训练集txt的文件夹名称
    parser.add_argument('--train_txt_folder', type=str, default='list/train_list.txt')
    # 输入验证集txt的文件夹名称
    parser.add_argument('--valid_txt_folder', type=str, default='list/valid_list.txt')
    # 数据集的父文件夹的路径
    parser.add_argument('--dataset_Parent_folder', type=str, default='')
    #输入图像的分辨率。
    parser.add_argument('--resolution', type=int, default=384)

    args = parser.parse_args()

    return args
def load_txt(txt_folder=None, parent_folder='', logger=None,txt_path=None):
    # 创建两个空列表，用于存储加载的视频路径和标签。
    tmp_pictures, tmp_labels = [], []

    # 获取指定目录下的所有文本文件名
    if txt_folder != None:
        txt_names = os.listdir(txt_folder)
        # 遍历每个文本文件。
        for txt_name in txt_names:
            # 打开当前文本文件。
            with open(os.path.join(txt_folder, txt_name), 'r') as f:
                # 读取文本文件中的每一行。
                pictures_names = f.readlines()
                # 遍历每一行。
                for i in pictures_names:
                    # 检查是否包含'landmarks'，如果是则跳过当前行
                    if i.find('landmarks') != -1:
                        continue
                    # 检查当前视频路径对应的目录是否为空，如果是则跳过当前行
                    # if len(os.listdir(i.strip().split()[0])) == 0:
                    # if len(os.listdir(os.path.join(parent_folder, i.strip().split()[0]))) == 0:

                    # 改为判断该图片文件是否存在
                    if not os.path.exists(os.path.join(parent_folder, i.strip().split()[0])):
                        continue

                    # 将视频路径和标签添加到临时列表中
                    # tmp_pictures.append(i.strip().split()[0])
                    tmp_pictures.append(os.path.join(parent_folder, i.strip().split()[0]))

                    #真假标签的位置,如果给定三个标签的话应该把最后的1改为3
                    tmp_labels.append(int(i.strip().split()[1]))
    else:
        with open(txt_path, 'r') as f:
            # 读取文本文件中的每一行。
            pictures_names = f.readlines()
            # 遍历每一行。
            for i in pictures_names:
                # 检查是否包含'landmarks'，如果是则跳过当前行
                if i.find('landmarks') != -1:
                    continue

                # 改为判断该图片文件是否存在
                if not os.path.exists(os.path.join(parent_folder, i.strip().split()[0])):
                    continue

                # 将视频路径和标签添加到临时列表中
                tmp_pictures.append(os.path.join(parent_folder, i.strip().split()[0]))
                # 真假标签的位置,如果给定三个标签的话应该把最后的1改为3
                tmp_labels.append(int(i.strip().split()[1]))

        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        # 打印当前加载的标签信息
        print(timeStr, len(tmp_labels), sum(tmp_labels), sum(tmp_labels)/len(tmp_labels))
    # 返回加载的视频路径列表和标签列表
    return tmp_pictures, tmp_labels



args = parse_args()
# some parameters
# model_name = 'MNIST'
model_name = 'deepfake'


# 根据输入图像，构建数据转换函数
# transform_train, transform_test = build_transforms(args.resolution, args.resolution,
#                                                    max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
#                                                    norm_std=[0.229, 0.224, 0.225])

train_pictures, train_labels = [], []
val_pictures, val_labels = [], []

# 这行代码调用了 load_txt 函数来加载额外的训练数据。他从 .txt 文件夹中加载包含视频路径和标签的txt文件。
tmp_train_pictures, tmp_train_labels = load_txt(txt_path=args.train_txt_folder,
                                                parent_folder=args.dataset_Parent_folder)
# 这两行代码将新加载的训练数据（视频路径和对应标签）添加到原始训练数据集的末尾。
tmp_val_pictures, tmp_val_labels = load_txt(txt_path=args.valid_txt_folder, parent_folder=args.dataset_Parent_folder)


# 训练集文件路径及标签
train_pictures += tmp_train_pictures
train_labels += tmp_train_labels

# 验证集文件路径及标签
val_pictures += tmp_val_pictures
val_labels += tmp_val_labels


# 1.0 Import the Dataset
dataset = Dataset(model_name, config_path='config.json',train_image_names=train_pictures,train_image_labels=train_labels,
                  test_image_names=val_pictures,test_image_labels=val_labels)
# dataset = Dataset(model_name, config_path='config.json')
# 1.1 Visualize imported dataset
# n_images = 2 # number of images to be plotted
# plotImages(dataset.X_test[:n_images,...,0], dataset.y_test[:n_images], n_images, dataset.class_names)

# 2.0 Load the Model
model_train = EfficientCapsNet(model_name, mode='train', verbose=True)

# 3.0 Train the Model
dataset_train, dataset_val = dataset.get_tf_data()
print(type(dataset_train))
# for inputs, labels in dataset_train:
#     print(f"Input tensor type: {type(inputs)}")
#     print(f"Label tensor type: {type(labels)}")
#     print(f"Input shape: {inputs.shape}")
#     print(f"Label shape: {labels.shape}")
#     break
history = model_train.train(dataset, initial_epoch=0)

plotHistory(history)