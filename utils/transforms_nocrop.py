from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import dlib

# 加载人脸检测器
detector = dlib.get_frontal_face_detector()


def detect_and_crop_face(image,**kwargs):
    # 将 NumPy 数组转换为 dlib 可处理的格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 进行人脸检测
    faces = detector(image, 1)

    # 如果检测到人脸
    if len(faces) > 0:
        # 取第一个检测到的人脸
        face = faces[0]
        # 裁剪人脸区域
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        width = x2 - x1
        height = y2 - y1

        # 将裁剪区域相比人脸扩大1.3倍
        Enlarged_size = 1.3
        mat = (Enlarged_size - 1) / 2

        x1 -= int(width * mat)
        y1 -= int(height * mat)
        x2 += int(width * mat)
        y2 += int(height * mat)
        # 确保裁剪区域不超出图像边界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        cropped_face = image[y1:y2, x1:x2]
        return cropped_face
    else:
        # 如果没有检测到人脸,返回原始图像
        return image
# 这个函数的作用是构建用于训练和测试过程中的图像数据转换（data transformation），确保输入模型的数据具有一致性和可训练性。这个函数接受以下参数：
# height 和 width：目标图像的高度和宽度。
# max_pixel_value：图像像素值的最大值，通常为255。
# norm_mean 和 norm_std：图像标准化的均值和标准差。如果未提供这些参数，函数将使用默认的 ImageNet 数据集的均值和标准差。
# **kwargs：其他可能的参数。
def build_transforms(height, width, max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
                     norm_std=[0.229, 0.224, 0.225], **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.E
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
        max_pixel_value (float): max pixel value
    """

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    # train_transform，用于训练集的数据转换。这个转换通常包括随机水平翻转、高斯噪声、高斯模糊、调整图像大小和颜色归一化等操作。最后，将图像转换为张量形式。
    train_transform = A.Compose([
        # A.Lambda(name="detect_and_crop_face", image=detect_and_crop_face),
        A.HorizontalFlip(p=0.2),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(p=0.1),
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])
    # test_transform：用于测试集的数据转换。这个转换通常只包括调整图像大小和颜色归一化，并将图像转换为张量形式。
    test_transform = A.Compose([
        # A.Lambda(name="detect_and_crop_face", image=detect_and_crop_face),
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])



    return train_transform, test_transform






