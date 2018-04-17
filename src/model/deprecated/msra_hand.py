import numpy as np
import matplotlib
import gc
# from data.transformations import transformPoints2D
# from util.handdetector import HandDetector
# from util.helpers import shuffle_many_inplace
from model.auto_encodermodel import ResEncoderModel
from model.hangregsolver import HandRegSolver
from model.poseregmodel import PoseRegModel
from model.resregmodel import ResNetModel
from model.solver import SolverParams, Solver
from util.handdetector import HandDetector

matplotlib.use('Agg')  # plot to file
# import matplotlib.pyplot as plt
# import os
# import pickle
import time
import sys
from data.importers import MSRA15Importer
from data.dataset import MSRA15Dataset
# from util.handpose_evaluation import MSRAHandposeEvaluation
from sklearn.decomposition import PCA

if __name__ == '__main__':

    rng = np.random.RandomState(23455)

    print("create data")
    aug_modes = ['com', 'rot', 'none']  # 'sc',

    comref = None  # "./eval/MSRA15_COM_AUGMENT/net_MSRA15_COM_AUGMENT.pkl"
    docom = False
    di = MSRA15Importer('../data/MSRA15/', refineNet=comref)
    seqs = []
    # seqs.append(di.loadSequence('P0', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P1', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P2', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P3', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P4', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P5', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P6', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P7', shuffle=True, rng=rng, docom=docom))
    # seqs.append(di.loadSequence('P8', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('Retrain', shuffle=True, rng=rng, docom=docom, cube=(200,200,200)))


    # dataSize = 0
    # for seq in seqs:
    #     for df in seq.data:
    #         dataSize+=df.dpt.nbytes
    #
    # print("data size {}mb".format(dataSize/1024/1024))

    testSeqs = [seqs[0]]
    # trainSeqs = [seq for seq in seqs if seq not in testSeqs]
    trainSeqs = [seqs[0]]

    print("training: {}".format(' '.join([s.name for s in trainSeqs])))
    print("testing: {}".format(' '.join([s.name for s in testSeqs])))

    # create training data
    trainDataSet = MSRA15Dataset(trainSeqs, localCache=False)
    nSamp = np.sum([len(s.data) for s in trainSeqs])
    d1, g1 = trainDataSet.imgStackDepthOnly(trainSeqs[0].name)  # 在这里进行归一化处理
    # 存储所有数据（[-1,1]）
    train_data = np.ones((nSamp, d1.shape[1], d1.shape[2], d1.shape[3]), dtype='float32')
    # 存储所有的标签([-1,1])
    train_gt3D = np.ones((nSamp, g1.shape[1], g1.shape[2]), dtype='float32')
    # 存储所有的立体限制框
    train_data_cube = np.ones((nSamp, 3), dtype='float32')
    # 存储所有的中心点
    train_data_com = np.ones((nSamp, 3), dtype='float32')
    # 存储所有的3D点
    train_gt3Dcrop = np.ones_like(train_gt3D)
    del d1, g1
    gc.collect()
    gc.collect()
    gc.collect()
    oldIdx = 0
    for seq in trainSeqs:
        d, g = trainDataSet.imgStackDepthOnly(seq.name)
        train_data[oldIdx:oldIdx + d.shape[0]] = d
        train_gt3D[oldIdx:oldIdx + d.shape[0]] = g
        train_data_cube[oldIdx:oldIdx + d.shape[0]] = np.asarray([seq.config['cube']] * d.shape[0])
        train_data_com[oldIdx:oldIdx + d.shape[0]] = np.asarray([da.com for da in seq.data])
        train_gt3Dcrop[oldIdx:oldIdx + d.shape[0]] = np.asarray([da.gt3Dcrop for da in seq.data])
        oldIdx += d.shape[0]
        del d, g
        gc.collect()
        gc.collect()
        gc.collect()

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    testDataSet = MSRA15Dataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly(testSeqs[0].name)
    test_data_cube = np.asarray([testSeqs[0].config['cube']] * test_data.shape[0])

    # 矩阵转置
    train_data = np.transpose(train_data, axes=[0, 2, 3, 1])
    test_data = np.transpose(test_data, axes=[0, 2, 3, 1])

    val_data = test_data
    val_gt3D = test_gt3D
    val_data_cube = test_data_cube

    print(train_gt3D.max(), test_gt3D.max(), train_gt3D.min(), test_gt3D.min())
    print(train_data.max(), test_data.max(), train_data.min(), test_data.min())

    # PCA 操作
    # pca = PCA(n_components=30)
    # pca.fit(train_gt3D.reshape(-1, train_gt3D.shape[1]*3))
    #
    # train_gt3D_embed = pca.transform(train_gt3D.reshape(-1, train_gt3D.shape[1]*3))
    # val_gt3D_embed = pca.transform(val_gt3D.reshape(-1, val_gt3D.shape[1]*3))
    # test_gt3D_embed = pca.transform(test_gt3D.reshape(-1, test_gt3D.shape[1]*3))

    aug_modes = ['com', 'rot', 'none']

    params = SolverParams(batchsize=64, printEvery=30, epochs=5, weight_decay=5e-3, optimizer='AdamOptimizer',
                          opt_params={'learning_rate': 1e-3}, aug_modes=aug_modes)
    X_data = {'train': train_data, 'val': val_data, 'test': test_data}
    # y_data = {'train': train_gt3D_embed, 'val': val_gt3D_embed, 'test': test_gt3D_embed}
    y_data = {'train': train_gt3D.reshape(-1, train_gt3D.shape[1]*3),
              'val': val_gt3D.reshape(-1, val_gt3D.shape[1]*3),
              'test': test_gt3D.reshape(-1, test_gt3D.shape[1]*3)}
    Cubes = {'train': train_data_cube, 'val': val_data_cube, 'test': test_data_cube}
    gt3Dcrops = {'train': train_gt3Dcrop}
    coms = {'train':train_data_com}
    solver = HandRegSolver(Xdata=X_data, Ydata=y_data, params=params,
                           Cubes=Cubes, pca=None, gt3Dcrops=gt3Dcrops,
                           hd=HandDetector(train_data[0,:,:,0].copy(), abs(di.fx), abs(di.fy), importer=di),
                           coms=coms)
    # model = PoseRegModel()
    model = ResEncoderModel(embedSize=20)
    solver.train(model, draw=False, cacheModel=True)

    # print('sleep 20s')
    # time.sleep(20)




