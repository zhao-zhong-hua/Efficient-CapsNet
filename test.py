# Efficient-CapsNet模型测试

import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages
from models import EfficientCapsNet
import time
import argparse
import os
# 设置GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)



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
# 一些参数
# model_name = 'MNIST'
model_name = 'deepfake'
custom_path = '/big-data/person/zhaozhonghua/capsnet/bin/efficient_capsnetdeepfake_new_train.h5'  # 如果您训练了新模型,请在此处插入完整的图权重路径

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


# 1.0 导入数据集
# dataset = Dataset(model_name, config_path='config.json')
dataset = Dataset(model_name, config_path='config.json',train_image_names=train_pictures,train_image_labels=train_labels,
                  test_image_names=val_pictures,test_image_labels=val_labels)
# 1.1 可视化导入的数据集
# n_images = 20  # 要绘制的图像数量
# plotImages(dataset.X_test[:n_images, ..., 0], dataset.y_test[:n_images], n_images, dataset.class_names)

# 2.0 加载模型
model_test = EfficientCapsNet(model_name, mode='test', verbose=True, custom_path=custom_path)
model_test.load_graph_weights()  # 加载图权重(bin文件夹)

# 3.0 测试模型
model_test.evaluate(dataset.X_test, dataset.y_test)  # 如果是"smallnorb"使用X_test_patch

# # 3.1 绘制分类错误的图像
# # 不适用于MultiMNIST
# y_pred = model_test.predict(dataset.X_test)[0]  # 如果是"smallnorb"使用X_test_patch
#
# n_images = 20
# plotWrongImages(dataset.X_test, dataset.y_test, y_pred,  # 如果是"smallnorb"使用X_test_patch
#                 n_images, dataset.class_names)