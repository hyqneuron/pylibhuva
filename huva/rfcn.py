from . import local_config
import sys
sys.path.append(local_config.rfcn_tools_folder)

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from pprint import pprint as pp
from PIL import Image

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    plt.show()

fff = '293'
path_fff = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/7{}_frames/'.format(fff)

def get_path(i):
    return path_fff + '{}_{:04d}.jpg'.format(fff, i)

def get_crop_prefix(i):
    return path_fff + 'crops/{}_{:04d}_'.format(fff, i)

def casy(net, image_name, show_det=True, output_prefix=None):
    """
    1. Load Image
    2. Detect all 'car'
    3. Crop detected cars and write output
    """
    # Load the demo image
    im = cv2.imread(image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES):
        if cls != 'car': continue # only look for car
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if show_det:
            vis_detections(im, cls, dets, thresh=CONF_THRESH)
            plt.show()
        for i in range(dets.shape[0]):
            score = dets[i][4]
            if score < CONF_THRESH: continue
            x1,y1,x2,y2 = int(dets[i][0]), int(dets[i][1]), int(dets[i][2]), int(dets[i][3])
            if output_prefix:
                cropped = im[y1:y2, x1:x2]
                cv2.imwrite(output_prefix + '{}.jpg'.format(i), cropped)

def get_car_bboxes(net, im):
    """
    1. Optionally load image if supplied im is string
    2. Detect all 'car'
    3. Return detected bounding boxes in (x1,y1,x2,y2)
    """
    if isinstance(im, str):
        im = cv2.imread(im)
    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    car_bboxes = []
    for cls_ind, cls in enumerate(CLASSES):
        if cls != 'car': continue # only look for car
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        for i in range(dets.shape[0]):
            score = dets[i][4]
            if score < CONF_THRESH: continue
            x1,y1,x2,y2 = int(dets[i][0]), int(dets[i][1]), int(dets[i][2]), int(dets[i][3])
            car_bboxes.append((x1, y1, x2, y2))
    return car_bboxes

def casy_batch():
    for i in range(1, 1147):
        casy(net, get_path(i), show_det=False, output_prefix=get_crop_prefix(i))

dirs = {
        '287':'/mnt/hdd3t/data/huva_cars/20170206-lornie-road/7287_frames',
        '288':'/mnt/hdd3t/data/huva_cars/20170206-lornie-road/7288_frames',
        '289':'/mnt/hdd3t/data/huva_cars/20170206-lornie-road/7289_frames',
        '290':'/mnt/hdd3t/data/huva_cars/20170206-lornie-road/7290_frames',
        }
merged_dir = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/merged_287_288_289_290/merged'

def show_img(np_img):
    # converts a np array image to PIL Image then show it. Better than matplotlib.pyplot.imshow
    rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(rgb_img))
    img.show()

import csv
from glob import glob

def read_csv(dir_name=None):
    """
    Returns filename_to_box: {'287_0102_0.jpg': [] or [x,y,width,height]}
    """
    if dir_name==None: dir_name = merged_dir
    csv_name = dir_name + '/via_region_data.csv'
    reader = csv.reader(open(csv_name))
    rows = [row for row in reader]
    rows = rows[1:]  # first line is header
    filenames = [fname.split('/')[-1] for fname in glob(dir_name + '/*.jpg')]
    filename_to_bbox = {fname:[] for fname in filenames}
    for row in rows:
        fname = row[0]
        num_bbox = int(row[3])
        if num_bbox == 1:
            bbox_str = row[5]
            words = bbox_str.split(';')[1:]
            # x, y, width, height
            numbers = [int(word.split('=')[1]) for word in words]
            filename_to_bbox[fname] = numbers
    return filename_to_bbox

def group_bboxes(filename_to_bbox):
    framename_to_bboxdict = {}
    for fname, bbox in filename_to_bbox.iteritems():
        head, frame_id, box_id = fname.split('.')[0].split('_')
        dir = dirs[head]
        framename_full = '{}/{}_{}.jpg'.format(dir, head, frame_id)
        if framename_full not in framename_to_bboxdict:
            framename_to_bboxdict[framename_full] = {}
        framename_to_bboxdict[framename_full][int(box_id)]  = bbox
    return framename_to_bboxdict

def get_overlapping_bboxes(car_bbox, plate_bboxes):
    cx1, cy1, cx2, cy2 = car_bbox
    overlaps = []
    for plate_bbox in plate_bboxes:
        px1, py1, px2, py2 = plate_bbox
        if px2 > cx1 and px1 < cx2 and py2 > cy1 and py1 < cy2:
            overlaps.append((px1 - cx1, py1 - cy1, px2 - cx1, py2 - cy1))
    return overlaps

def clip(val, minval, maxval):
    return max(minval, min(maxval, val))

def process_frame(framename_full, framename_to_bboxdict, net, save_prefix):
    im = cv2.imread(framename_full)
    bboxdict = framename_to_bboxdict[framename_full]
    # consolidate things into car_bboxes and plate_bboxes, each a [(x1,y1,x2,y2)]
    car_bboxes = get_car_bboxes(net, im)
    plate_bboxes = []
    im_H = im.shape[0]
    im_W = im.shape[1]
    for i, carbbox in enumerate(car_bboxes):
        carx1, cary1, carx2, cary2 = carbbox
        if i in bboxdict:
            if len(bboxdict[i]) == 0: continue
            platex1, platey1, platewidth, plateheight = bboxdict[i]
            platex1 += carx1
            platey1 += cary1
            platex2 = platex1 + platewidth
            platey2 = platey1 + plateheight
            plate_bboxes.append((platex1, platey1, platex2, platey2))
    # for each car_bbox, generate a larger crop, identify the plate_bboxes that overlap within, and write out
    """
    fix, ax = plt.subplots(figsize=(12,12))
    ax.imshow(im[:, :, [2,1,0]], aspect='equal')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    """
    crop_info = []
    for i,car_bbox in enumerate(car_bboxes):
        x1, y1, x2, y2 = car_bbox
        width = x2 - x1
        height= y2 - y1
        width_inc = width / 2
        height_inc = height / 2
        x1_new = clip(x1 - width_inc, 0, im_W)
        y1_new = clip(y1 - height_inc, 0, im_H)
        x2_new = clip(x2 + width_inc, 0, im_W)
        y2_new = clip(y2 + height_inc, 0, im_H)
        width_new  = x2_new - x1_new
        height_new = y2_new - y1_new
        # crop
        im_crop = im[y1_new:y2_new, x1_new: x2_new, :].copy()
        overlaps = get_overlapping_bboxes([x1_new, y1_new, x2_new, y2_new], plate_bboxes)
        jet = im_crop.copy()
        jet[:,:] = [255, 0, 0] # set to blue
        # heatbox
        heatboxes = []
        for overlap in overlaps:
            x1_o, y1_o, x2_o, y2_o = overlap
            x1_o = clip(x1_o, 0, width_new)
            y1_o = clip(y1_o, 0, height_new)
            x2_o = clip(x2_o, 0, width_new)
            y2_o = clip(y2_o, 0, height_new)
            heatboxes.append((x1_o, y1_o, x2_o, y2_o))
            jet[y1_o:y2_o, x1_o:x2_o, :] = [0, 0, 255] # set to red
        # show/save the crop
        fname = framename_full.split('/')[-1].split('.')[0]
        savename = '{}/{}_{}.jpg'.format(save_prefix, fname, i)
        #cv2.imwrite(savename, im_crop)
        crop_info.append((savename, heatboxes))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow((im_crop/2 + jet/2)[:,:,[2,1,0]], aspect='equal')
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        print(x1_new, y1_new, x2_new, y2_new)
    plt.show()
    return crop_info


import pickle
def align_stuffs_all():
    filename_to_bbox = read_csv()
    framename_to_bboxdict = group_bboxes(filename_to_bbox)
    names = framename_to_bboxdict.keys()
    crops_info = []
    for i,framename_full in enumerate(names):
        crops_info.append(process_frame(framename_full, framename_to_bboxdict, net, 
            '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/merged_287_288_289_290/bigcrops'))
        print(i)
    save_crops_info_path = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/merged_287_288_289_290/crops_info.pkl'
    pickle.dump(crops_info, open(save_crops_info_path, 'wb'))
    return crops_info
    
        

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    args = parser.parse_args()

    return args

net = None

def make(force=False):
    global net
    if force or net is None:
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        args = parse_args()

        prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                                'rfcn_end2end', 'test_agnostic.prototxt')
        caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                                  NETS[args.demo_net][1])

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\n').format(caffemodel))

        if args.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(net, im)

