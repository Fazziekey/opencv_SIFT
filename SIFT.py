# Developer：Fazzie
# Time: 2020/6/1420:25
# File name: SIFT.py
# Development environment: Anaconda Python

import cv2
import numpy as np
import scipy
import _pickle as pickle
import random
import os
from matplotlib import pyplot as plt
import scipy.spatial

# 特征提取模块
def feature_extract(image_path,vector_size=32):
    img = cv2.imread(image_path,1)#读取图片
    # 显示图片
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # SIFT特征提取
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    # 画出特征点并保存
    img = cv2.drawKeypoints(gray, kp, img)
    cv2.imwrite('sift_keypoints.jpg', img)

    # 根据关键点的返回值进行排序（越大越好）
    kp = sorted(kp, key=lambda x: -x.response)[:vector_size]

    # 计算描述符向量
    kp, des = sift.compute(gray, kp)

    # 将其放在一个大的向量中，作为我们的特征向量
    des = des.flatten()

    # 使描述符的大小一致
    # 描述符向量的大小为128
    needed_size = (vector_size * 128)
    if des.size < needed_size:
        # 如果少于32个描述符，则在特征向量后面补零
        des = np.concatenate([des, np.zeros(needed_size - des.size)])

    return des

#   数据存储
def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = feature_extract(f)

    # 将特征向量存于pickled 文件
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)

# 图像特征匹配模块
class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path,"rb") as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # 计算待搜索图像与数据库图像的余弦距离
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = feature_extract(image_path)
        img_distances = self.cos_cdist(features)
        # 获得前5个记录
        nearest_ids = np.argsort(img_distances)[:topn].tolist()

        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist()


def show_img(path):
    img = cv2.imread(path,1)
    plt.imshow(img)
    plt.show()

# 主程序
def run():
    images_path = 'E:/Code/SFIT/101_ObjectCategories/panda/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # 随机获取1张图

    sample = random.sample(files, 1)

    batch_extractor(images_path)

    ma = Matcher('features.pck')

    for s in sample:
        print('Query image ==========================================')
        show_img(s)
        names, match = ma.match(s, topn=3)
        print('Result images ========================================')
        for i in range(3):

            # 我们得到了余弦距离，向量之间的余弦距离越小表示它们越相似，因此我们从1中减去它以得到匹配值
            print('Match %s' % (1 - match[i]))
            show_img(os.path.join(images_path, names[i]))

run()



