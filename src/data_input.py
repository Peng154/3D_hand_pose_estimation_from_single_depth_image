import os
from PIL import Image
import numpy as np
from scipy import ndimage,stats
import cv2
from collections import namedtuple
import pickle

ICVLFrame = namedtuple('ICVLFrame', ['img', 'jointOrig', 'T', 'joint3DOrig', 'joint3DCrop', 'com', 'fileName'])

class DataInput(object):
    def __init__(self, dataPath, useCache , cacheDir, fx, fy, ux, uy):
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.dataPath = dataPath
        # 这个相机标定参数不是很懂啊！！！！
        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy


    def loadDepthImage(self, path):
        # 打开文件
        image = Image.open(path)
        # 确保是深度图
        assert len(image.getbands())==1
        imgData = np.asarray(image, np.float32)
        return imgData

    def jointsImgTo3D(self, joints):
        """
        将整只手的关节点图片坐标转换成3D坐标
        :param joints: numJoints*3 维度的数组
        :return: numJoints*3 的3D坐标数组
        """
        temp = np.zeros((joints.shape[0], 3), np.float32)
        for i in range(joints.shape[0]):
            temp[i] = self.jointImgTo3D(joints[i])
        return temp

    def jointImgTo3D(self, joint):
        '''
        这个东西和相机标定和坐标轴转换有关，先抄着
        :param joint: 需要转换的单个手指关节坐标 x，y是像素，z是mm
        :return:返回 xyz都是mm单位的3D坐标
        '''
        temp = np.zeros((3,), dtype=np.float32)
        temp[0] = (joint[0] - self.ux) * joint[2] / self.fx
        temp[1] = (joint[1] - self.uy) * joint[2] / self.fy
        temp[2] = joint[2]
        return temp

    def checkImage(self, img, stddev=1):
        """
        利用标准差来检查图片是否有手部等内容
        :param img:需要检查的图片
        :param stddev:图片标准差
        :return:
        """
        if np.std(img)<stddev:
            return False
        else:
            return True

    def calculateCom(self, img):
        """
        跟据图片来计算中心的位置
        :param img: 输入深度图
        :return: 返回计算得到的手部中心的坐标（xyz），z是mm单位
        """
        temp = img.copy()
        maxDepth = min(1500, temp.max())
        minDepth = max(10, temp.min())
        # 无关的数据设为0，背景以及太近的数据
        temp[temp>maxDepth] = 0
        temp[temp<minDepth] = 0
        center = ndimage.measurements.center_of_mass(temp)
        num = np.count_nonzero(temp)
        com = np.array((center[1]*num, center[0]*num, temp.sum()), dtype=np.float32)

        if num==0:
            return np.array((0, 0, 0), dtype=np.float32)
        else:
            # 算出深度的平均值之后返回
            return com/num

    def calculateBounds(self, com, cubeSize):
        """
        根据手部的中心点来计算整只手的3D边界的具体范围
        :param com: 手部中心（xyz）z单位mm，xy像素
        :param cubeSize: 3D边界的大小（xyz）单位mm
        :return: xyz的开始和结束位置 xy像素，z单位mm
        """
        zstart = com[2] - cubeSize[2] / 2.
        zend = com[2] + cubeSize[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / self.fx - cubeSize[0] / 2.) / com[2] * self.fx))
        xend = int(np.floor((com[0] * com[2] / self.fx + cubeSize[0] / 2.) / com[2] * self.fx))
        ystart = int(np.floor((com[1] * com[2] / self.fy - cubeSize[1] / 2.) / com[2] * self.fy))
        yend = int(np.floor((com[1] * com[2] / self.fy + cubeSize[1] / 2.) / com[2] * self.fy))
        return xstart, xend, ystart, yend, zstart, zend

############################################################
        # 这里看的不是很懂。。。。。
    def getNDValue(self, img):
        """
        Get value of not defined depth value distances
        :return:value of not defined depth value
        """
        maxDepth = min(1500, img.max())
        minDepth = max(10, img.min())

        if img[img < minDepth].shape[0] > img[img > maxDepth].shape[0]:
            return stats.mode(img[img < minDepth])[0][0]
        else:
            return stats.mode(img[img > maxDepth])[0][0]
        # 不懂结束
#############################################################


    def cropHand3DArea(self, img, com=None, cubeSize=(250, 250, 250), dsize=(128,128)):
        """

        :param com: 手掌的中心位置，用于作为截取图像的位置参考,(x,y,z)其中z的单位是mm，xy是像素
        :param cubeSize: 需要截取的3D空间的大小，单位是mm
        :param dsize: 截取之后图像结果的大小，单位是像素
        :return:返回截取得到的手部图像，变换矩阵
        """
        maxDepth = min(1500, img.max())
        minDepth = max(10, img.min())
        # 把无关的深度设为0，这里挺重要的。。。。。
        img[img > maxDepth] = 0.
        img[img < minDepth] = 0.

        # 确保截取空间的正确性
        if not (len(cubeSize)==3 and len(dsize)==2):
            raise ValueError('截取的一定是3D空间或者输出图片一定是二维的')
        # 如果没有中心，需要计算中心，但是可能连续识别的时候需要改一下。。。。。
        if com is None:
            com = self.calculateCom(img=img)
        # 计算边界
        xstart, xend, ystart, yend, zstart, zend = self.calculateBounds(com, cubeSize)
##############################################################
        # 开始截取图片
        cropped = img[max(ystart, 0):min(yend, img.shape[0]), max(xstart, 0):min(xend, img.shape[1])].copy()
        # 补全缺失的像素
        cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                       abs(yend) - min(yend, img.shape[0])),
                                      (abs(xstart) - max(xstart, 0),
                                       abs(xend) - min(xend, img.shape[1]))), mode='constant', constant_values=0)

        # 如果深度小于z开始值，置为zstart
        mask1 = np.bitwise_and(cropped < zstart, cropped != 0)
        # 如果深度大于z开始值，置为0（因为背景目前数值是0）
        mask2 = np.bitwise_and(cropped > zend, cropped != 0)
        cropped[mask1] = zstart
        cropped[mask2] = 0.

############################################################
        # 这里看的不是很懂。。。。。又是相机坐标转换
        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = np.asmatrix(np.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        if cropped.shape[0] > cropped.shape[1]:
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[1] / float(cropped.shape[0]))
        else:
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[0] / float(cropped.shape[1]))
        scale[2, 2] = 1

        sz2 = (int(sz[0]), int(sz[1]))
        rz = cv2.resize(cropped, sz2, interpolation=cv2.INTER_NEAREST)

        # pylab.imshow(rz); pylab.gray();t=transformPoint2D(com,scale*trans);pylab.scatter(t[0],t[1]); pylab.show()
        ret = np.ones(dsize, np.float32) * self.getNDValue(img=img)  # use background as filler
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz

        off = np.asmatrix(np.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, off * scale * trans

        # 不懂结束
#############################################################


    def normalizeData(self, dataFrames, size=(250,250,250)):
        """
        将图片和手指关节点（label）归一化到[-1,1]
        :param dataFrames:ICVLFrame格式的列表
        :return:imgs, labels (numpy.array)
        """
        nums = len(dataFrames)
        imgs = np.zeros((nums, 1, dataFrames[0].img.shape[0], dataFrames[0].img.shape[1]), dtype=np.float32)
        labels = np.zeros((nums, dataFrames[0].joint3DCrop.shape[0], dataFrames[0].joint3DCrop.shape[1]), dtype=np.float32)

        for i in range(nums):
            imgD = np.asarray(dataFrames[i].img.copy(), 'float32')
            # print(imgD.max(), " ", imgD.min(), " ", imgSeq.data[i].com[2])
            imgD[imgD == 0] = dataFrames[i].com[2] + (size[2] / 2.)
            imgD -= dataFrames[i].com[2]
            imgD /= (size[2] / 2.)

            imgs[i] = imgD
            labels[i] = np.clip(np.asarray(dataFrames[i].joint3DCrop, dtype='float32') / (size[2] / 2.), -1, 1)

        return imgs, labels

class TCVLDataInput(DataInput):
    def __init__(self, dataPath='../data/ICVL', useCache=True, cacheDir='./cache'):
        super().__init__(dataPath, useCache, cacheDir, 241.42, 241.42, 160., 120.)
        # 手指关节点的数目
        self.numJoints = 16
        self.config = {'cube':[250,250,250], 'dsize':(128, 128)}

    def loadData(self, dataName, shuffle=True, rng=None):
        """
        加载截取后的手部图像，以及手部关节（相对于中心）标签
        全部都归一化到了[-1,1]
        :param dataName: 需要加载的数据名称
        :param shuffle: 是否打乱
        :param rng: 随机种子
        :return: np.array类型的imgs，labels
        """

        if rng is None:
            rng = self.rng = np.random.RandomState(23455)

        labelFilePath = '{}/{}.txt'.format(self.dataPath, dataName)
        cacheFilePath = '{}/{}.pkl'.format(self.cacheDir, dataName)

        print('开始从{}加载数据。。。。'.format(labelFilePath))

        print(cacheFilePath)

        # 如果使用缓存并且已经有缓存文件存在，直接返回
        if self.useCache:
            if os.path.exists(cacheFilePath):
                f = open(cacheFilePath, 'rb')
                data,self.config = pickle.load(f)
                f.close()

                if shuffle and rng is not None:
                    print("Shuffling")
                    rng.shuffle(data)

                # 将图片和标签规范化到[-1,1]
                imgs, labels = self.normalizeData(data, size=self.config['cube'])
                return imgs, labels

        objsDir = '{}/{}'.format(self.dataPath, 'Depth')
        labelsFile = open(labelFilePath)
        # 移动指针到0位置
        labelsFile.seek(0)

        data = []

        for line in labelsFile:
            parts = line.split(' ')
            imgPath = '{}/{}'.format(objsDir, parts[0])
            if not os.path.isfile(imgPath):
                print("File {} is not existed!".format(imgPath))
                continue

            # 得到深度图
            imgData = self.loadDepthImage(imgPath)
            # 得到手指关节坐标点
            jointsOrg = np.zeros((self.numJoints, 3),dtype=np.float32)
            for i in range(self.numJoints):
                for j in range(3):
                    jointsOrg[i,j] = parts[3*i + j + 1]

            # 转换成原始的3D坐标
            joints3DOrg = self.jointsImgTo3D(jointsOrg)

            #如果图片内容为空
            if not self.checkImage(imgData, 1):
                print("File {} is no content!".format(imgPath))
                continue

            #不为空，提取手部部分
            try:
                imgData, M = self.cropHand3DArea(imgData, jointsOrg[0], cubeSize=self.config['cube'], dsize=self.config['dsize'])
            except UserWarning:
                print("Skipping image {}, no hand detected".format(imgPath))
                continue

            com3D = self.jointImgTo3D(jointsOrg[0])
            # 手指关节点归一化到质心
            joints3Dcrop = joints3DOrg - com3D
            data.append(ICVLFrame(imgData.astype(np.float32), jointsOrg, M, joints3DOrg, joints3Dcrop, com3D, imgPath))

        labelsFile.close()
        print("Loaded {} samples.".format(len(data)))
        if self.useCache:
            print("Save cache data to {}".format(cacheFilePath))
            f = open(cacheFilePath, 'wb')
            pickle.dump((data,self.config), f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        # 打乱顺序
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)

        # 将图片和标签规范化到[-1,1]
        imgs, labels = self.normalizeData(data, size=self.config['cube'])

        return imgs,labels


