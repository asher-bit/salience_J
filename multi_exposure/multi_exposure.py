import cv2
import os
import numpy as np
import time
import math

###### generating multi-exposure ldr images from hdr images
###### created by nzk (eezkni@gmail.com)

def img_resize(img, min_length=720):
    """
    Resize the image proportionally with a minimum side length of min_length
    :param img: input image with three channels
    :param min_length: minimum side length of image after resize
    :return:
    """
    w, h, c = img.shape
    min_length = np.min([w, h, min_length])
    if w <= h:
        img_resize = cv2.resize(img, (int(min_length * h / w), min_length), cv2.INTER_AREA)
    else:
        img_resize = cv2.resize(img, (min_length, int(min_length * w / h)), cv2.INTER_AREA)
    return img_resize

def create_dir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)

def map_range_ldr(x, mode='norm'):
    if mode == 'norm':
        return (x / 255.0).astype('float32')
    elif mode == 'to_uint8':
        return (x * 255).astype('uint8')

def get_curves(filename):
    # == get the camera curve
    with open(filename, 'r') as f:
        lines = f.readlines()
    I = []  # used in the loop
    B = []
    I_list = []  # all the data
    B_list = []
    num_curves = 0  # count
    # plt.figure()
    for i in range(len(lines)):
        if (lines[i] == 'I =\n'):
            I = [float(i) for i in lines[i + 1].split()]
        if (lines[i] == 'B =\n'):
            B = [float(i) for i in lines[i + 1].split()]
        if I and B:
            # plt.plot(I, B)
            I_list.append(I)
            B_list.append(B)
            I = []
            B = []
            num_curves += 1
    return I_list, B_list

def _pre_hdr_norm(hdr):
    hdr_mean = np.mean(hdr)
    hdr = 0.5 * hdr / (hdr_mean + 1e-6)
    return hdr

def param(hdr_max, hdr_median, num_ev=15, dist=7.):
    middle = 0  # use middle=0
    start = middle - dist
    step = np.ones(int(num_ev)) * (middle - start) / ((num_ev - 1) / 2)

    return start, step

def create_EV(data_path, save_path, index_curve, num_ev=15, dist=7.0, resize=False):
    # == get the camera curve
    filename = r'/home/jiaoshengjie/Saliency_Code/Saliency_transformer/multi_exposure/dorfCurves.txt'
    I_list, B_list = get_curves(filename)
    create_dir(save_path)
    ldr_type = '.png'
    data_extension = ['.exr', '.hdr', 'png']
    num = 0

    for root, dirs, files in sorted(os.walk(data_path)):
        for file in files:
            if any(
                    file.lower().endswith(extension)
                    for extension in data_extension
            ):
                hdr_path = os.path.join(data_path, file)
                hdr_name = file
                num += 1

                for i in range(len(index_curve)):
                    print('Processing {}/{}, {}, curve: {}'.format(num, len(files), hdr_name, index_curve[i]))

                    # hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYCOLOR + cv2.IMREAD_ANYDEPTH)
                    hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
                    if resize:
                        hdr = img_resize(hdr, 1024)
                    hdr = _pre_hdr_norm(hdr)

                    start, step = param(hdr.max(), np.median(hdr), num_ev=9, dist=dist) # use median = 0
                    
                    #exp_times = [1/4, 1/(2 * math.sqrt(2)), 1/2, 1/(math.sqrt(2)), 1, math.sqrt(2), 2, 2 * math.sqrt(2), 4]
                    for idx in range(0, 9):
                        ind = index_curve[i]
                        I, B = I_list[ind], B_list[ind]

                        exp_time = start + step[0] * idx
                        # exp_time = exp_times[idx]
                        delta = 2.0 ** (0.5 * exp_time)  

                        ldr = hdr * delta
                        np.clip(ldr, 0, 1)
                        ldr = np.interp(ldr, I, B).astype(ldr.dtype)
                        ldr = map_range_ldr(ldr, mode='to_uint8').astype('uint8')

                        cv2.imwrite(os.path.join(save_path, hdr_name[:-4] + '_crf' + str(ind) + '_' + chr(97 + idx) + ldr_type), ldr)
                       

if __name__ == '__main__':
    # selected curve for dataset creating
    # ind = [166, 154, 71, 16, 114, 63, 74]
    # ind = [154, 16, 114, 63, 74]
    ind = [71]
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    #hdr_path = r'/data/jiaoshengjie/IMLHDR/image_mantiuk'
    hdr_path = r'/data/jiaoshengjie/IMLHDR/imlhdr_dataset'
    #ev_path = r'/data/jiaoshengjie/IMLHDR/multi_exposure'
    ev_path = r'/data/jiaoshengjie/IMLHDR/multi_exposure'

    start_time = time.time()

    create_EV(hdr_path, ev_path, ind, num_ev=15, dist=7, resize=False)

    current_time = time.time()
    duration = current_time - start_time

    print("=== Time used [{:>.4f} mins] ===".format(duration/60))

