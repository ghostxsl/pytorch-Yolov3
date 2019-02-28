import sys
import time
from PIL import Image, ImageDraw
import torch
from utils import *
from model import YoLoNet
from torch.utils.data import DataLoader
import dataset
from torchvision import datasets, transforms

anchors1 = [116, 90, 156, 198, 373, 326]
for i in range(len(anchors1)):
    anchors1[i] = anchors1[i] / 32
anchors2 = [30,61,  62,45,  59,119]
for i in range(len(anchors2)):
    anchors2[i] = anchors2[i] / 16
anchors3 = [10,13,  16,30,  33,23]
for i in range(len(anchors3)):
    anchors3[i] = anchors3[i] / 8
anchors = [anchors1,anchors2,anchors3]

batch_size = 4
num_workers = 0
gpus = '0'
init_width = 416
init_height = 416
conf_thresh = 0.7
nms_thresh = 0.4
iou_thresh = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%s" % gpus if use_cuda else "cpu")
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

def dotest(weightfile, testfile, namesfile, outfile):
    model = YoLoNet(20,3)
    model.width = init_width
    model.height = init_height
    test_loader = DataLoader(
        dataset.listDataset(testfile, shape=(model.width, model.height),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]), test_txt=True,
                            imgdirpath='../VOC/VOC2012_train/JPEGImages/',
                            labdirpath='../VOC/VOC2012_train/Labels/'),
        batch_size=batch_size, shuffle=False, **kwargs)

    class_names = load_class_names(namesfile)

    model = model.to(device)
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    model.eval()

    num_classes = model.num_classes
    num_anchors = model.num_anchors

    f = open(outfile, 'a')
    with torch.no_grad():
        for batch_idx, (data, filename) in enumerate(test_loader):
            data = data.to(device)
            output1, output2, output3 = model(data)
            Out = [output1, output2, output3]
            all_boxes = []
            nBoxes = []
            for i in range(data.size(0)):
                nBoxes.append([])
            for i in range(len(Out)):
                all_boxes.append(get_region_boxes(Out[i], conf_thresh, num_classes, anchors[i], num_anchors, device=device))

            for i in range(data.size(0)):
                for j in range(3):
                    nBoxes[i] += all_boxes[j][i]

            for i in range(data.size(0)):
                boxes = nBoxes[i]
                if len(boxes) == 0:
                    continue
                boxes = nms(boxes, nms_thresh)

                for j in range(len(boxes)):
                    f.write(class_names[int(boxes[j][6].item())] + ' ' + filename[i] + ' ' + '%0.6f' % (boxes[j][4].item()) + ' ' + '%0.6f' %
                                (boxes[j][0]*init_width) + ' ' + '%0.6f' % (boxes[j][1]*init_height)
                                + ' ' + '%0.6f' % (boxes[j][2] * init_width)+ ' ' + '%0.6f' % (boxes[j][3]*init_height))
    f.close()


if __name__ == '__main__':
    if len(sys.argv) == 5:
        weightfile = sys.argv[1]
        testfile = sys.argv[2]
        namesfile = sys.argv[3]
        outfile = sys.argv[4]
        dotest(weightfile, testfile, namesfile, outfile)
    else:
        print('Usage: ')
        print('  python test_txt.py weightfile testfile namesfile outfile')
