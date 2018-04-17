from multiprocessing import Process
import numpy as np
from collections import namedtuple

DataBatch = namedtuple('DataBatch', ['imgs', 'gt3Dcrops', 'cubes', 'coms'])

class DataBatchProducer(Process):
    def __init__(self, indices, rawData, batchSize, hd, aug_modes, queue):
        super().__init__()
        # print(type(indices))
        self.indices = indices
        self.rawData = rawData
        self.bathSize = batchSize
        self.hd = hd
        self.queue = queue
        self.aug_modes = aug_modes
        self.rng = np.random.RandomState(6666)

    def run(self):
        start_idx = 0
        while True:
            idxs = self.indices[start_idx:start_idx+self.bathSize]

            pre_imgs = self.rawData.imgs[idxs]
            pre_3Dcrops = self.rawData.gt3Dcrops[idxs]
            pre_coms = self.hd.importer.joints3DToImg(self.rawData.coms[idxs])
            pre_cubes = self.rawData.cubes[idxs]

            imgs = np.zeros_like(pre_imgs, dtype=np.float32)
            gt3Dcrops = np.zeros_like(pre_3Dcrops, dtype=np.float32)
            coms = np.zeros_like(pre_coms, dtype=np.float32)
            cubes = np.zeros_like(pre_cubes, dtype=np.float32)

            for i in range(pre_imgs.shape[0]):
                imgs[i,:,:,0], gt3Dcrops[i], cubes[i], coms[i],_ = self.augmentCrop(pre_imgs[i,:,:,0], pre_3Dcrops[i],
                                                                           pre_coms[i], pre_cubes[i], np.eye(3),
                                                                           self.aug_modes, self.hd)

            self.queue.put(DataBatch(imgs, gt3Dcrops, cubes, coms))
            start_idx = (start_idx + self.bathSize)%self.indices.shape[0]

    def augmentCrop(self, img, gt3Dcrop, com, cube, M, aug_modes, hd,  sigma_com=None,
                    sigma_sc=None):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :return: image, 3D annotations, com, cube
        """

        assert len(img.shape) == 2
        assert isinstance(aug_modes, list)

        if sigma_com is None:
            sigma_com = 5.

        if sigma_sc is None:
            sigma_sc = 0.02

        img = img * (cube[2] / 2.) + com[2]
        premax = img.max()

        mode = self.rng.randint(0, len(aug_modes))
        off = self.rng.randn(3) * sigma_com  # +-px/mm
        rot = self.rng.uniform(0, 360)
        sc = abs(1. + self.rng.randn() * sigma_sc)
        if aug_modes[mode] == 'com':
            imgD, new_joints3D, com, M = hd.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, pad_value=0)
            curLabel = new_joints3D / (cube[2] / 2.)
        elif aug_modes[mode] == 'rot':
            imgD, new_joints3D, _ = hd.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, pad_value=0)
            curLabel = new_joints3D / (cube[2] / 2.)
        elif aug_modes[mode] == 'sc':
            imgD, new_joints3D, cube, M = hd.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, pad_value=0)
            curLabel = new_joints3D / (cube[2] / 2.)
        elif aug_modes[mode] == 'none':
            imgD = img
            curLabel = gt3Dcrop / (cube[2] / 2.)
        else:
            raise NotImplementedError()

        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)

        return imgD, curLabel, np.asarray(cube), com, M


