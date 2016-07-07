"""
Enable custom dataset training in py-faster-rcnn
by fulfilling mimimal funcionality of creating a roidb

__author__ = "Joshua Zhang"
"""

import os
from datasets.imdb import imdb
import numpy as np
import uuid
import scipy.sparse
from PIL import Image
import cPickle
from fast_rcnn.config import cfg

class CustomDB(imdb):
    def __init__(self, image_set, name, root_path='custom'):
        imdb.__init__(self, 'custom_' + name + '_' + image_set)
        self._image_set = image_set
        self._root_path = os.path.join(cfg.DATA_DIR, root_path)
        self._data_path = os.path.join(self._root_path, name)
        self._image_path = os.path.join(self._root_path, name, 'images')
        self._label_path = os.path.join(self._root_path, name, 'labels')
        self._classes = ('__background__', # always index 0
                        '0', '1', '2', '3', '4')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._salt = str(uuid.uuid4())

        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._root_path), \
               'Custom root path does not exist: {}'.format(self._root_path)
        assert os.path.exists(self._data_path), \
               'Data path does not exist: {}'.format(self._data_path)
        assert os.path.exists(self._image_path), \
               'Image path does not exist: {}'.format(self._image_path)
        assert os.path.exists(self._label_path), \
               'Label path does not exist: {}'.format(self._label_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._image_path, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_custom_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_custom_annotation(self, index):
        """
        Loading image and bounding boxes info from customized formats
        Modify here according to your need
        This version cooperate with YOLO label format(id, xmin, xmax, ymin, ymax)
        """
        imname = os.path.join(self._image_path, index + self._image_ext)
        labelname = os.path.join(self._label_path, index + '.txt')
        boxes = []
        gt_classes = []
        overlaps = []
        seg_areas = []
        with open(labelname, 'rt') as f:
            for line in f:
                cls_index, x, y, w, h = line.split()
                cls_index = int(cls_index) + 1 # background + 1
                # Unfortunately we have to get the image size cause
                # annotations don't have it
                img = Image.open(imname)
                width, height= img.size
                xmin = (float(x) - float(w)/2) * width -1
                ymin = (float(y) - float(h)/2) * height - 1
                xmax = (float(x) + float(w)/2) * width - 1
                ymax = (float(y) + float(h)/2) * height - 1
                boxes.append([xmin, ymin, xmax, ymax])
                gt_classes.append(cls_index)
                tmp = [0.0] * self.num_classes
                tmp[cls_index] = 1.0
                overlaps.append(tmp)
                seg_areas.append((xmax - xmin + 1) * (ymax - ymin + 1))

        boxes = np.array(boxes).astype(np.uint16)
        gt_classes = np.array(gt_classes).astype(np.int32)
        overlaps = np.array(overlaps).astype(np.float32)
        seg_areas = np.array(seg_areas).astype(np.float32)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def rpn_roidb(self):
        raise NotImplementedError
