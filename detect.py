import sys
import time
from PIL import Image, ImageDraw
import torch
from utils import *
from model import YoLoNet
from config import Config

def detect(weightfile, imgfile, yolo_config, GTfile=None):
    model = YoLoNet(yolo_config.num_classes, yolo_config.in_channels)

    model.load_weights(weightfile)
    model.height = 416
    model.width = 416
    print('Loading weights from %s... Done!' % (weightfile))

    if yolo_config.num_classes == 20:
        namesfile = 'data/voc.names'
    elif yolo_config.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda: 0 " if use_cuda else "cpu")

    model = model.to(device)

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((model.width, model.height))

    start = time.time()
    boxes = do_detect(model, sized, 0.5, 0.4, yolo_config.get_anchors(), device)
    if boxes is None:
        print('No boxes!!!!')
        return None
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names, GTfile)
    print('Done!')

if __name__ == '__main__':
    if len(sys.argv) == 3:
        weightfile = sys.argv[1]
        imgfile = sys.argv[2]
        GTfile = None
        yolo_config = Config()
        detect(weightfile, imgfile, yolo_config, GTfile)
    else:
        print('Usage: ')
        print('  python detect.py weightfile imgfile')
        # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
