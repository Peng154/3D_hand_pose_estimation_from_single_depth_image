"""
This is the main file for testing realtime performance.

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy, os
import src.run_testing as test
from src.config import FLAGS
from src.model.cpm_hand_model import CPM_Model
from src.data.dataset import NYUDataset, ICVLDataset
from src.util.realtimehandposepipeline import RealtimeHandposePipeline
from src.data.importers import ICVLImporter, NYUImporter, MSRA15Importer
from src.util.cameradevice import CreativeCameraDevice, FileDevice, DepthSenseCameraDevice

if __name__ == '__main__':
    rng = numpy.random.RandomState(23455)

    di = MSRA15Importer('../data/MSRA15/')
    # Seq2 = di.loadSequence('P0')
    # testSeqs = [Seq2]

    # di = ICVLImporter('../data/ICVL/')
    # Seq2 = di.loadSequence('test_seq_1')
    # testSeqs = [Seq2]

    # di = NYUImporter('../data/NYU/')
    # Seq2 = di.loadSequence('test_1')
    # testSeqs = [Seq2]

    # load trained network
    # config = {'fx': 241.42, 'fy': 241.42, 'cube': (200, 200, 200)} # msra_file
    config = {'fx': 365.26, 'fy': 365.26, 'cube': (250, 250, 250)} # kinect

    model_path_suffix = '{}_{}_stage{}'.format(FLAGS.model_name, FLAGS.data_set, FLAGS.stages)
    model_weights_path = os.path.join(FLAGS.cacheDir,
                                      FLAGS.weightDir,
                                      model_path_suffix,
                                      '{}-{}'.format(FLAGS.model_name, FLAGS.test_iters))

    model = test.get_model(model_weights_path)

    # model = ResEncoderModel(n_dim=63, cacheFile='../msra_res_encoder_model/model.cpkt')
    # model = ResEncoderModel(n_dim=63, embedSize=20, cacheFile='../train_cache/model.cpkt')
    rtp = RealtimeHandposePipeline(model, config, di, verbose=False, comrefNet=None)

    # use filenames
    # filenames = []
    # for i in testSeqs[0].data:
    #     filenames.append(i.fileName)
    # dev = FileDevice(filenames, di)

    # use depth camera
    dev = DepthSenseCameraDevice()
    rtp.processVideoThreaded(dev, draw3D=False, sendToUnity=True)
