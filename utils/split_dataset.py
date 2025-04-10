"""
# split dataset
#
"""

import os
import glob
import random
from PIL import Image

if __name__ == '__main__':
    split_rate = 0.2  # 0.8:0.2
    resize_image = 224  # 图片缩放后统一大小
    file_path = 'E:\Papers\dataset_spatNet\\flower_photos'  # 获取原始数据集路径
    train_dir = 'E:\Papers\dataset_spatNet\\train'
    test_dir = 'E:\Papers\dataset_spatNet\\test'
    # 找到文件中所有文件夹的目录，即类文件夹名
    dirs = glob.glob(os.path.join(file_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    print("Totally {} classes: {}".format(len(dirs), dirs))  # 打印花类文件夹名称

    for path in dirs:
        # 对每个类别进行单独处理
        path = path.split('\\')[-1]  # -1表示以分隔符/保留后面的一段字符

        os.makedirs(os.path.join(train_dir, path), exist_ok=True)
        os.makedirs(os.path.join(test_dir, path), exist_ok=True)

        # 读取原始数据集中path类中对应类型的图片，并添加到files中
        files = glob.glob(os.path.join(file_path, path, '*jpg'))
        files += glob.glob(os.path.join(file_path, path, '*jpeg'))
        files += glob.glob(os.path.join(file_path, path, '*png'))

        random.shuffle(files)  # 打乱图片顺序
        split_boundary = int(len(files) * split_rate)  # 训练集和测试集的划分边界

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')

            # 更改原始图片尺寸
            old_size = img.size  # (wight, height)
            ratio = float(resize_image) / max(old_size)  # 通过最长的size计算原始图片缩放比率
            # 把原始图片最长的size缩放到resize_pic，短的边等比率缩放，等比例缩放不会改变图片的原始长宽比
            new_size = tuple([int(x * ratio) for x in old_size])

            im = img.resize(new_size, Image.Resampling.LANCZOS)  # 更改原始图片的尺寸，并设置图片高质量，保存成新图片im
            new_im = Image.new("RGB", (resize_image, resize_image))  # 创建一个resize_pic尺寸的黑色背景
            # 把新图片im贴到黑色背景上，并通过'地板除//'设置居中放置
            new_im.paste(im, ((resize_image - new_size[0]) // 2, (resize_image - new_size[1]) // 2))

            # 先划分0.1_rate的测试集，剩下的再划分为0.9_rate的训练集，同时直接更改图片后缀为.jpg
            assert new_im.mode == "RGB"
            if i < split_boundary:
                new_im.save(os.path.join(test_dir, path,
                                         file.split('\\')[-1].split('.')[0] + '.jpg'))
            else:
                new_im.save(os.path.join(train_dir, path,
                                         file.split('\\')[-1].split('.')[0] + '.jpg'))

    # 统计划分好的训练集和测试集中.jpg图片的数量
    print("Train directory:", train_dir)
    print("Test directory:", test_dir)
    train_files = glob.glob(os.path.join(train_dir, '*', '*.jpg'))
    test_files = glob.glob(os.path.join(test_dir, '*', '*.jpg'))

    print("Totally {} files for train".format(len(train_files)))
    print("Totally {} files for test".format(len(test_files)))
