
import cPickle

default_crops_info_path =  '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/merged_287_288_289_290/crops_info.pkl'
def read_crops_info(filepath=None):
    """
    read crops_info from pickle output by demo_rfcn.py in github/py_rfcn
    crops_info : [(str, [(x1,y2,x2,y2)])]
        where str is crop_full_path
        x1,y2,x2,y2 form a bounding box for heat
    """
    filepath = filepath or default_crops_info_path
    with open(filepath, 'rb') as f:
        crops_info = cPickle.load(f)
    crops_info = [pic_info for frame_info in crops_info for pic_info in frame_info]
    return crops_info
