"""Class for running the handpose estimation pipeline in realtime.

RealtimeHandposePipeline provides interface for running the pose estimation.
It is made of detection, image cropping and further pose estimation.

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

import copy
import time
import tensorflow as tf
import socket
from collections import deque
from ctypes import c_bool
from multiprocessing import Process, Manager, Value, Pool

import cv2
import numpy
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from data.transformations import rotatePoints3D
from util.handdetector import HandDetector
from util.handpose_evaluation import ICVLHandposeEvaluation, NYUHandposeEvaluation, MSRAHandposeEvaluation


class RealtimeHandposePipeline(object):
    """
    Realtime pipeline for handpose estimation
    """

    # states of pipeline
    STATE_IDLE = 0
    STATE_INIT = 1
    STATE_RUN = 2

    # different hands
    HAND_LEFT = 0
    HAND_RIGHT = 1

    # different detectors
    DETECTOR_COM = 0

    def __init__(self, poseNet, config, di, verbose=False, comrefNet=None):
        """
        Initialize data
        :param poseNet: network for pose estimation
        :param config: configuration
        :param di: depth importer
        :param verbose: print additional info
        :param comrefNet: refinement network from center of mass detection
        :return: None
        """

        # handpose CNN
        self.importer = di
        self.poseNet = poseNet
        self.comrefNet = comrefNet
        # configuration
        self.initialconfig = copy.deepcopy(config)
        # synchronization between processes
        self.sync = Manager().dict(config=config, fid=0,
                                   crop=numpy.ones((128, 128), dtype='float32'),
                                   com3D=numpy.asarray([0, 0, 300]),
                                   frame=numpy.ones((240, 320), dtype='float32'), M=numpy.eye(3))
        self.start_prod = Value(c_bool, False)
        self.start_con = Value(c_bool, False)
        self.stop = Value(c_bool, False)
        # for calculating FPS
        self.lastshow = time.time()
        self.runningavg_fps = deque(100*[0], 100)
        self.verbose = verbose
        # hand left/right
        self.hand = Value('i', self.HAND_LEFT)
        # initial state
        self.state = Value('i', self.STATE_IDLE)
        # detector
        self.detection = Value('i', self.DETECTOR_COM)
        # hand size estimation
        self.handsizes = []
        self.numinitframes = 50
        # hand tracking or detection
        self.tracking = Value(c_bool, False)
        self.lastcom = (0, 0, 0)
        # show different results
        self.show_pose = False
        self.show_crop = True

    def initNets(self):
        """
        Init network in current process
        :return: 
        """
        # Force network to compile output in the beginning
        # self.output, _ = self.poseNet.inference(self.poseNet.X, self.poseNet.y)\
        self.output = self.poseNet.model_output
        self.sess = tf.Session()
        saver = tf.train.Saver()
        import os
        print('模型绝对路径：',os.path.abspath(self.poseNet.cacheFile))
        if(os.path.exists(self.poseNet.cacheFile+'.index')):
            print('模型存在。。。。。。。。。。。。。。。。。。。')
        else:
            print('模型不存在。。。。。。。。。。。。。。。。。')
        # 加载模型参数
        saver.restore(self.sess, self.poseNet.cacheFile)
        print('加载模型成功。。。。。。。。。。。')

        # if self.comrefNet is not None:
        #     if isinstance(self.comrefNet, ScaleNetParams):
        #         self.comrefNet = ScaleNet(numpy.random.RandomState(23455), cfgParams=self.comrefNet)
        #         self.comrefNet.computeOutput([numpy.zeros(sz, dtype='float32') for sz in self.comrefNet.cfgParams.inputDim])
        #     else:
        #         raise RuntimeError("Unknown refine method!")

    def threadProducer(self, device):
        """
        Thread that produces frames from video capture
        :param device: device
        :return: None
        """
        device.start()
        # self.initNets()
        # Nets are compiled, ready to run
        self.start_prod.value = True

        fid = 0
        while True:
            if self.start_con.value is False:
                time.sleep(0.1)
                continue

            if self.stop.value is True:
                break
            # Capture frame-by-frame
            start = time.time()
            ret, frame = device.getDepth()
            if ret is False:
                print("Error while reading frame.")
                time.sleep(0.1)
                continue
            if self.verbose is True:
                print("{}ms capturing".format((time.time() - start)*1000.))

            startd = time.time()
            crop, M, com3D = self.detect(frame.copy())
            if self.verbose is True:
                print("{}ms detection".format((time.time() - startd)*1000.))

            self.sync.update(fid=fid, crop=crop, com3D=com3D, frame=frame, M=M)
            fid += 1

        # we are done
        print("Exiting producer...")
        device.stop()
        return True

    def threadConsumer(self, tmp):
        """
        Thread that consumes the frames, estimate the pose and display
        :return: None
        """

        self.initNets()
        # Nets are compiled, ready to run
        self.start_con.value = True
        while True:
            if self.start_prod.value is False:
                time.sleep(0.1)
                continue

            if self.stop.value is True:
                break

            frm = copy.deepcopy(self.sync)

            startp = time.time()
            pose = self.estimatePose(frm['crop'], frm['com3D'])
            pose = pose * self.sync['config']['cube'][2]/2. + frm['com3D']
            if self.verbose is True:
                print("{}ms pose".format((time.time() - startp)*1000.))

            # Display the resulting frame
            starts = time.time()
            img, poseimg = self.show(frm['frame'], pose)
            img = self.addStatusBar(img)
            cv2.imshow('frame', img)
            self.lastshow = time.time()
            if self.show_pose:
                cv2.imshow('pose', poseimg)
            if self.show_crop:
                cv2.imshow('crop', numpy.clip((frm['crop'] + 1.)*128., 0, 255).astype('uint8'))
            self.processKey(cv2.waitKey(1) & 0xFF)
            if self.verbose is True:
                print("{}ms display".format((time.time() - starts)*1000.))

        cv2.destroyAllWindows()
        # we are done
        print("Exiting consumer...")
        return True

    def a(self, num):
        for i in range(num):
            print(i)
            self.i = i

    def drawHandPoseIn3D(self, ax, pose):
        """
        利用matplotlib展示手的3D框架
        :param pose:
        :return:
        """
        print(pose.shape)

        plt.pause(0.001)
        pass

    def processVideoThreaded(self, device, draw3D=False, sendToUnity=False):
        """
        Use video as input
        :param device: device id
        :return: None
        """

        # print("Create producer process...")
        # p = Process(target=self.threadProducer, args=[device])
        # p.daemon = True
        # print("Create consumer process...")
        # c = Process(target=self.threadConsumer, args=[1])
        # c.daemon = True
        # p.start()
        # c.start()
        #
        # c.join()
        # p.join()
        device.start()
        self.initNets()
        fid = 0

        ax = None
        lines = None
        if draw3D:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-100,600)
            ax.set_ylim(-500,200)
            ax.set_zlim(300,800)

            lines = []

        # 状态 0：未开始， 1：开始记录， 2：记录结束
        status = 0
        frames = []
        fps = 0
        record_start_time = None

        if sendToUnity:
            # socket client
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 获取本地主机名
            host = "127.0.0.1"
            # 设置端口号
            port = 5354
            s.connect((host, port))

        while True:

            # 获取帧
            if self.stop.value is True:
                break
            # Capture frame-by-frame
            start = time.time()
            ret, frame = device.getDepth()
            if ret is False:
                print("Error while reading frame.")
                time.sleep(0.1)
                continue
            if self.verbose is True:
                print("{}ms capturing".format((time.time() - start)*1000.))

            startd = time.time()
            crop, M, com3D = self.detect(frame.copy())
            if self.verbose is True:
                print("{}ms detection".format((time.time() - startd)*1000.))

            self.sync.update(fid=fid, crop=crop, com3D=com3D, frame=frame, M=M)
            fid += 1

            # print(crop.shape)
            # cv2.imshow('crop', crop)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('c'):
                if status == 0:
                    print('开始记录数据。')
                    status = 1
                elif status == 1:
                    print('记录数据结束。')
                    status = 2

            # 消费帧
            frm = copy.deepcopy(self.sync)

            startp = time.time()
            pose = self.estimatePose(frm['crop'], frm['com3D'])
            pose = pose * self.sync['config']['cube'][2] / 2. + frm['com3D']

            # 手反了，要转过来？？？？我当初写的是啥？？？
            if pose.shape[0] == 16:
                cube_y = numpy.array([frm['com3D'][1]]*pose.shape[0])
                pose[:, 1] = 2 * cube_y - pose[:,1]

            # 发送数据帧
            if sendToUnity:
                if pose.shape[0] == 21:
                    # 把大拇指放在前面
                    temp = numpy.zeros_like(pose, dtype=numpy.float32)
                    temp[0] = pose[0]
                    temp[1:5] = pose[17:21]
                    temp[5:21] = pose[1:17]
                elif pose.shape[0] == 16:
                    temp = numpy.array(pose, dtype=numpy.float32)
                flat = temp.flatten().tolist()
                flat = [str(element) for element in flat]
                msg = ';'.join(flat)
                # print(msg)
                s.send(msg.encode("utf-8"))

            # 记录数据帧
            if status == 1:
                if key == ord('c'):
                    record_start_time = time.time()
                frames.append(pose)
            elif status == 2:
                if key == ord('c'):
                    fps = len(frames) / (time.time() - record_start_time)
                    print("{}fps is recorded".format(fps))
                    import pickle
                    file = open('frames.pkl', 'wb')
                    pickle.dump([fps, frames],file)

            if self.verbose is True:
                print("{}ms pose".format((time.time() - startp) * 1000.))

            if draw3D:
                assert lines is not None and ax is not None
                # 画3D点
                # 清除旧的手
                # print('3D--------------')
                for l in lines:
                    ax.collections.remove(l)
                lines.clear()
                x = pose[:,0]
                y = pose[:,1]
                z = pose[:,2]

                lineX = numpy.ones(shape=(2), dtype=numpy.float32)
                lineY = numpy.ones(shape=(2), dtype=numpy.float32)
                lineZ = numpy.ones(shape=(2), dtype=numpy.float32)

                dim = pose.shape[0]
                finger_joints_num = (dim - 1) // 5
                count = -1
                colors = ['b','g','r','y','k']
                for i in range(1,dim):
                    s = -1
                    if i % finger_joints_num == 1:
                        s = 0
                        count += 1
                    else:
                        s = i-1
                    lineX[0], lineY[0], lineZ[0] = x[s], y[s], z[s]
                    lineX[1], lineY[1], lineZ[1] = x[i], y[i], z[i]
                    lines.append(ax.plot_wireframe(lineX, lineY, lineZ, rstride=2, cstride=2, color=colors[count]))

                plt.pause(0.001)

            # Display the resulting frame
            starts = time.time()
            img, poseimg = self.show(frm['frame'], pose)
            img = self.addStatusBar(img)
            cv2.imshow('frame', img)
            self.lastshow = time.time()
            if self.show_pose:
                cv2.imshow('pose', poseimg)
            if self.show_crop:
                cv2.imshow('crop', numpy.clip((frm['crop'] + 1.) * 128., 0, 255).astype('uint8'))
            self.processKey(cv2.waitKey(1) & 0xFF)
            if self.verbose is True:
                print("{}ms display".format((time.time() - starts) * 1000.))

        cv2.destroyAllWindows()
        device.stop()

    def processVideo(self, device):
        """
        Use video as input
        :param device: device id
        :return: None
        """
        device.start()

        self.initNets()

        i = 0
        while True:
            i += 1
            if self.stop.value is True:
                break
            # Capture frame-by-frame
            start = time.time()
            ret, frame = device.getDepth()
            if ret is False:
                print("Error while reading frame.")
                time.sleep(0.1)
                continue
            if self.verbose is True:
                print("{}ms capturing".format((time.time() - start)*1000.))

            startd = time.time()
            crop, M, com3D = self.detect(frame.copy())
            if self.verbose is True:
                print("{}ms detection".format((time.time() - startd)*1000.))

            startp = time.time()
            pose = self.estimatePose(crop, com3D)
            pose = pose*self.sync['config']['cube'][2]/2. + com3D
            if self.verbose is True:
                print("{}ms pose".format((time.time() - startp)*1000.))

            # Display the resulting frame
            starts = time.time()
            img, poseimg = self.show(frame, pose)

            img = self.addStatusBar(img)
            cv2.imshow('frame', img)
            self.lastshow = time.time()
            if self.show_pose:
                cv2.imshow('pose', poseimg)
            if self.show_crop:
                cv2.imshow('crop', numpy.clip((crop + 1.)*128., 0, 255).astype('uint8'))
            self.processKey(cv2.waitKey(1) & 0xFF)
            if self.verbose is True:
                print("{}ms display".format((time.time() - starts)*1000.))
                print("-> {}ms per frame".format((time.time() - start)*1000.))

        # When everything done, release the capture
        cv2.destroyAllWindows()
        device.stop()

    def detect(self, frame):
        """
        Detect the hand
        :param frame: image frame
        :return: cropped image, transformation, center
        """

        hd = HandDetector(frame, self.sync['config']['fx'], self.sync['config']['fy'], importer=self.importer, refineNet=self.comrefNet)
        doHS = (self.state.value == self.STATE_INIT)
        if self.tracking.value and not numpy.allclose(self.lastcom, 0):
            loc, handsz = hd.track(self.lastcom, self.sync['config']['cube'], doHandSize=doHS)
        else:
            loc, handsz = hd.detect(size=self.sync['config']['cube'], doHandSize=doHS)

        self.lastcom = loc

        if self.state.value == self.STATE_INIT:
            self.handsizes.append(handsz)
            if self.verbose is True:
                print(numpy.median(numpy.asarray(self.handsizes), axis=0))
        else:
            self.handsizes = []

        if self.state.value == self.STATE_INIT and len(self.handsizes) >= self.numinitframes:
            cfg = self.sync['config']
            cfg['cube'] = tuple(numpy.median(numpy.asarray(self.handsizes), axis=0).astype('int'))
            self.sync.update(config=cfg)
            self.state.value = self.STATE_RUN
            self.handsizes = []

        if numpy.allclose(loc, 0):
            return numpy.zeros((self.poseNet.input_images.get_shape()[1].value,
                                self.poseNet.input_images.get_shape()[2].value),
                               dtype='float32'), numpy.eye(3), loc
        else:
            crop, M, com = hd.cropArea3D(com=loc, size=self.sync['config']['cube'],
                                         dsize=(self.poseNet.input_images.get_shape()[1].value,
                                                self.poseNet.input_images.get_shape()[2].value))
            com3D = self.importer.jointImgTo3D(com)
            sc = (self.sync['config']['cube'][2] / 2.)
            crop[crop == 0] = com3D[2] + sc
            crop.clip(com3D[2] - sc, com3D[2] + sc)
            crop -= com3D[2]
            crop /= sc
            return crop, M, com3D

    def estimatePose(self, crop, com3D):
        """
        Estimate the hand pose
        :param crop: cropped hand depth map
        :param com3D: com detection crop position
        :return: joint positions
        """

        # mirror hand if left/right changed
        if self.hand.value == self.HAND_LEFT:
            inp = crop[None, :, :, None].astype('float32')
        else:
            inp = crop[None, :, ::-1, None].astype('float32')

        jts = self.sess.run(self.output, feed_dict={self.poseNet.input_images:inp})
        jj = jts[0].reshape((-1, 3))

        if 'invX' in self.sync['config']:
            if self.sync['config']['invX'] is True:
                # mirror coordinates
                jj[:, 1] *= (-1.)

        if 'invY' in self.sync['config']:
            if self.sync['config']['invY'] is True:
                # mirror coordinates
                jj[:, 0] *= (-1.)

        # mirror pose if left/right changed
        if self.hand.value == self.HAND_RIGHT:
            # mirror coordinates
            jj[:, 0] *= (-1.)
        return jj

    def show(self, frame, handpose):
        """
        Show depth with overlaid joints
        :param frame: depth frame
        :param handpose: joint positions
        :return: image
        """
        upsample = 1.
        if 'upsample' in self.sync['config']:
            upsample = self.sync['config']['upsample']

        # plot depth image with annotations
        imgcopy = frame.copy()
        # display hack to hide nd depth
        msk = numpy.logical_and(32001 > imgcopy, imgcopy > 0)
        msk2 = numpy.logical_or(imgcopy == 0, imgcopy == 32001)
        min = imgcopy[msk].min()
        max = imgcopy[msk].max()
        imgcopy = (imgcopy - min) / (max - min) * 255.
        imgcopy[msk2] = 255.
        imgcopy = imgcopy.astype('uint8')
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)

        if not numpy.allclose(upsample, 1):
            imgcopy = cv2.resize(imgcopy, dsize=None, fx=upsample, fy=upsample, interpolation=cv2.INTER_LINEAR)

        if handpose.shape[0] == 16:
            hpe = ICVLHandposeEvaluation(numpy.zeros((3, 3)), numpy.zeros((3, 3)))
        elif handpose.shape[0] == 14:
            hpe = NYUHandposeEvaluation(numpy.zeros((3, 3)), numpy.zeros((3, 3)))
        elif handpose.shape[0] == 21:
            hpe = MSRAHandposeEvaluation(numpy.zeros((3, 3)), numpy.zeros((3, 3)))
        else:
            raise ValueError("Invalid number of joints {}".format(handpose.shape[0]))

        jtI = self.importer.joints3DToImg(handpose)
        jtI[:, 0:2] -= numpy.asarray([frame.shape[0]//2, frame.shape[1]//2])
        jtI[:, 0:2] *= upsample
        jtI[:, 0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])
        for i in range(handpose.shape[0]):
            cv2.circle(imgcopy, (jtI[i, 0], jtI[i, 1]), 3, (255, 0, 0), -1)

        for i in range(len(hpe.jointConnections)):
            cv2.line(imgcopy, (jtI[hpe.jointConnections[i][0], 0], jtI[hpe.jointConnections[i][0], 1]),
                     (jtI[hpe.jointConnections[i][1], 0], jtI[hpe.jointConnections[i][1], 1]),
                     255.*hpe.jointConnectionColors[i], 2)

        # comI = self.importer.joint3DToImg(com3D)
        # comI[0:2] -= numpy.asarray([frame.shape[0]//2, frame.shape[1]//2])
        # comI[0:2] *= upsample
        # comI[0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])
        # cv2.circle(imgcopy, (comI[0], comI[1]), 3, (0, 255, 0), 1)

        poseimg = numpy.zeros_like(imgcopy)
        # rotate 3D pose and project to 2D
        jtP = self.importer.joints3DToImg(rotatePoints3D(handpose, handpose[self.importer.crop_joint_idx], 0., 90., 0.))
        jtP[:, 0:2] -= numpy.asarray([frame.shape[0]//2, frame.shape[1]//2])
        jtP[:, 0:2] *= upsample
        jtP[:, 0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])
        for i in range(handpose.shape[0]):
            cv2.circle(poseimg, (jtP[i, 0], jtP[i, 1]), 3, (255, 0, 0), -1)

        for i in range(len(hpe.jointConnections)):
            cv2.line(poseimg, (jtP[hpe.jointConnections[i][0], 0], jtP[hpe.jointConnections[i][0], 1]),
                     (jtP[hpe.jointConnections[i][1], 0], jtP[hpe.jointConnections[i][1], 1]),
                     255.*hpe.jointConnectionColors[i], 2)

        # comP = self.importer.joint3DToImg(rotatePoint3D(com3D, handpose[self.importer.crop_joint_idx], 0., 90., 0.))
        # comP[0:2] -= numpy.asarray([frame.shape[0]//2, frame.shape[1]//2])
        # comP[0:2] *= upsample
        # comP[0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])
        # cv2.circle(poseimg, (comP[0], comP[1]), 3, (0, 255, 0), 1)

        return imgcopy, poseimg

    def addStatusBar(self, img):
        """
        Add status bar to image
        :param img: image
        :return: image with status bar
        """
        barsz = 20
        retimg = numpy.ones((img.shape[0]+barsz, img.shape[1], img.shape[2]), dtype='uint8')*255

        retimg[barsz:img.shape[0]+barsz, 0:img.shape[1], :] = img

        # FPS text
        fps = 1./(time.time()-self.lastshow)
        self.runningavg_fps.append(fps)
        avg_fps = numpy.mean(self.runningavg_fps)
        cv2.putText(retimg, "FPS {0:2.1f}".format(avg_fps), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand text
        cv2.putText(retimg, "Left" if self.hand.value == self.HAND_LEFT else "Right", (80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand size
        ss = "HC-{0:d}".format(self.sync['config']['cube'][0])
        cv2.putText(retimg, ss, (120, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand tracking mode, tracking or detection
        cv2.putText(retimg, "T" if self.tracking.value else "D", (260, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand detection mode, COM or CNN
        if self.detection.value == self.DETECTOR_COM:
            mode = "COM"
        else:
            mode = "???"
        cv2.putText(retimg, mode, (280, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # status symbol
        if self.state.value == self.STATE_IDLE:
            col = (0, 0, 255)
        elif self.state.value == self.STATE_INIT:
            col = (0, 255, 255)
        elif self.state.value == self.STATE_RUN:
            col = (0, 255, 0)
        else:
            col = (0, 0, 255)
        cv2.circle(retimg, (5, 5), 5, col, -1)
        return retimg

    def processKey(self, key):
        """
        Process key
        :param key: key value
        :return: None
        """

        if key == ord('q'):
            self.stop.value = True
        elif key == ord('h'):
            if self.hand.value == self.HAND_LEFT:
                self.hand.value = self.HAND_RIGHT
            else:
                self.hand.value = self.HAND_LEFT
        elif key == ord('+'):
            cfg = self.sync['config']
            cfg['cube'] = tuple([lst + 10 for lst in list(cfg['cube'])])
            self.sync.update(config=cfg)
        elif key == ord('-'):
            cfg = self.sync['config']
            cfg['cube'] = tuple([lst - 10 for lst in list(cfg['cube'])])
            self.sync.update(config=cfg)
        elif key == ord('r'):
            self.reset()
        elif key == ord('i'):
            self.state.value = self.STATE_INIT
        elif key == ord('t'):
            self.tracking.value = not self.tracking.value
        elif key == ord('s'):
            self.show_crop = not self.show_crop
            self.show_pose = not self.show_pose
        else:
            pass

    def reset(self):
        """
        Reset stateful parts
        :return: None
        """
        self.state.value = self.STATE_IDLE
        self.sync.update(config=copy.deepcopy(self.initialconfig))
        self.detection.value = self.DETECTOR_COM
