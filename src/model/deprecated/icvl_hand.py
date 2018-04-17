import numpy as np
import matplotlib
import gc
# from data.transformations import transformPoints2D
# from util.handdetector import HandDetector
# from util.helpers import shuffle_many_inplace
from src.model.auto_encodermodel import ResEncoderModel
from src.model.hangregsolver import HandRegSolver
from src.model.poseregmodel import PoseRegModel
from src.model.resregmodel import ResNetModel
from src.model.solver import SolverParams, Solver
from src.util.handdetector import HandDetector

matplotlib.use('Agg')  # plot to file
import time
import sys
from src.data.importers import ICVLImporter
from src.data.dataset import ICVLDataset
# from util.handpose_evaluation import MSRAHandposeEvaluation
from sklearn.decomposition import PCA

if __name__ == '__main__':
    rng = np.random.RandomState(23455)

    print("create data")
    aug_modes = ['com', 'rot', 'none']  # 'sc',

    comref = None  # "./eval/ICVL_COM_AUGMENT/net_ICVL_COM_AUGMENT.pkl"
    docom = False
    di = ICVLImporter('../data/ICVL/', refineNet=comref)
    Seq1 = di.loadSequence('train', shuffle=True, rng=rng, docom=docom)
    # Seq1 = di.loadSequence('train',subSeq=['0'], shuffle=True, rng=rng, docom=docom)
    trainSeqs = [Seq1]

    Seq2 = di.loadSequence('test_seq_1')
    testSeqs = [Seq2]

    # create training data
    trainDataSet = ICVLDataset(trainSeqs)
    train_data, train_gt3D = trainDataSet.imgStackDepthOnly('train')
    train_data_cube = np.asarray([Seq1.config['cube']] * train_data.shape[0], dtype='float32')
    train_data_com = np.asarray([d.com for d in Seq1.data], dtype='float32')
    train_gt3Dcrop = np.asarray([d.gt3Dcrop for d in Seq1.data], dtype='float32')

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    # valDataSet = ICVLDataset(testSeqs)
    # val_data, val_gt3D = valDataSet.imgStackDepthOnly('test_seq_1')

    testDataSet = ICVLDataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly('test_seq_1')
    test_data_cube = np.asarray([Seq2.config['cube']] * test_data.shape[0], dtype='float32')

    # 矩阵转置
    train_data = np.transpose(train_data, axes=[0, 2, 3, 1])
    test_data = np.transpose(test_data, axes=[0, 2, 3, 1])
    print('train data shape:',train_data.shape)
    print('test data shape:',test_data.shape)

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

    # params = SolverParams(batchsize=128, printEvery=100, epochs=1, weight_decay=5e-3, optimizer='AdamOptimizer',
    #                       opt_params={'learning_rate': 1e-4}, aug_modes=aug_modes)
    params = SolverParams(batchsize=128, printEvery=100, epochs=1, weight_decay=5e-3, optimizer='AdamOptimizer',
                          opt_params={'learning_rate': 1e-5}, aug_modes=None)
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
    model = ResEncoderModel(n_dim=48)
    solver.train(model, draw=False, cacheModel=True)

    # print('sleep 20s')
    # time.sleep(20)
