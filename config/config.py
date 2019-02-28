import math
import numpy as np
import os


class Config(object):
    trainList = "train.txt"
    validationList = "val.txt"
    testList = "test.txt"
    imgDirPath = "../VOC/VOC2012_train/JPEGImages/"
    labelDirPath = "../VOC/VOC2012_train/Labels/"

    # Weight path or none
    weightFile = "none"
    # Loss visualization
    # ON or OFF
    tensorboard = False
    logsDir = "runs"

    # Train params

    # model save path
    backupDir = "backup"
    max_epochs = 500
    save_interval = 50
    # e.g. 0,1,2,3
    gpus = "0"
    # multithreading
    num_workers = 2
    batch_size = 8

    # Solver params
    # adma or sgd
    solver = "adam"
    steps = [8000, 16000]
    scales = [0.1, 0.1]
    learning_rate = 1e-5
    momentum = 0.9
    decay = 5e-4
    betas = (0.9, 0.98)

    # YoloNet params

    num_classes = 20
    in_channels = 3
    init_width = 416
    init_height = 416
    anchors1 = [116, 90, 156, 198, 373, 326]
    anchors2 = [30, 61, 62, 45, 59, 119]
    anchors3 = [10, 13, 16, 30, 33, 23]

    # def __init__(self):

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def get_anchors(self):
        anchor1 = []
        anchor2 = []
        anchor3 = []
        for i in range(len(self.anchors1)):
            anchor1.append(self.anchors1[i] / 32)
        for i in range(len(self.anchors2)):
            anchor2.append(self.anchors2[i] / 16)
        for i in range(len(self.anchors3)):
            anchor3.append(self.anchors3[i] / 8)
        anchors = [anchor1, anchor2, anchor3]
        return anchors