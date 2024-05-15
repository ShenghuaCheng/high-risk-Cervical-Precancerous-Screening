# -*- coding:utf-8 -*-
"""
this scripts is to read a low resolution WSI which is as:

A_LOW_RESOLUTION_WSI_DIR_PATH/
    |-- image1.jpg
    |-- iamge2.jpg
    ...

"""
import glob
import os
import cv2
import multiprocessing.dummy as mp
import multiprocessing


class SmallWsiRead(object):
    """
    to read a low resolution WSI
    """
    def __init__(self, wsi_dir_path):
        self.path_on_linux = wsi_dir_path
        self.support_format = ('bmp', 'jpg', 'tif')

        self.block_paths = glob.glob(os.path.join(self.path_on_linux, '*.jpg'))
        self.block_paths += glob.glob(os.path.join(self.path_on_linux, '*.bmp'))
        self.block_paths += glob.glob(os.path.join(self.path_on_linux, '*.tif'))
        self.block_num = len(self.block_paths)

        assert self.block_num > 0, '{}, block num = 0'.format(self.path_on_linux)

        self.instances_num = None
        self.instances = None

    def read(self):
        """
        slide -> [instance, ...]
        """
        self.instances = {}
        lock = multiprocessing.Lock()

        def block2instances(path: str):
            """
            center crop a view of slide to 1600 * 1600 pixels
            redundantly crop instance (256 * 256 pxiels) by step 192
            """
            block = cv2.imread(path)[:, :, ::-1]
            block = self.crop_preprocess(block)
            step = 192
            for ww in range(8):
                for hh in range(8):
                    name = path.split(os.sep)[-1].split('.')[0] + '_{}_{}'.format(ww * step, hh * step)
                    with lock:
                        self.instances[name] = block[hh * step: hh * step + 256, ww * step: ww * step + 256, :]

        pool = mp.Pool()
        for block_path in self.block_paths:
            pool.apply_async(block2instances, (block_path,))
        pool.close()
        pool.join()

        self.instances_num = len(self.instances)
        assert self.instances_num > 0, 'instance num = 0 !'

    @staticmethod
    def crop_preprocess(im):
        """
        center crop a view of a slide
        :param:im numpy image
        :return:im cropped numpy image
        """
        try:
            h, w, _ = im.shape
            im = im[h // 2 - 800: h // 2 + 800, w // 2 - 800: w // 2 + 800, :]
        except:
            print('IMG CORP ERROR')
            raise AssertionError

        return im

    def get_instance(self, block_path, w, h, size):
        """
        read region
        """
        block = cv2.imread(os.path.join(self.path_on_linux, block_path + '.' + self.format))[:, :, ::-1]
        block = self.crop_preprocess(block)
        instance = block[h: h + size, w: w + size, :]

        return instance
