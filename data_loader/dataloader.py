import io
import os
import cv2
import json
import random
import contextlib
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data_loader.data_augment import random_perspective, box_candidates, TrainTransform


class COCODatset(Dataset):
    def __init__(self ,cfg , img_size = (320 ,320),
                 name = "VOC2012" ,
                 json_file = "data.json",
                 preproc=None ,
                 no_aug = True , tracking = False ,
                 logger = None ):
        super(COCODatset, self).__init__()

        self.cgf = cfg
        self.img_size = img_size
        self.json_file = json
        self.preproc = preproc
        self.augment = not no_aug
        self.tracking = tracking
        self.logger = logger
        self.data_dir = self.cgf.data_dir
        self.batch_size = self.cgf.batch_size


        ##**************************************##
        ## Data Augment Params                  ##
        ##**************************************##
        self.random_size = self.cfg.random_size
        self.degree = self.cfg.degree
        self.translate = self.cfg.translate
        self.scale = self.cfg.scale
        self.shear = self.cfg.shear
        self.prespective = self.cfg.prespective
        self.mixup_scale = (0.5 , 1.5)
        self.enable_mosaic = self.cfg.enable_mixup
        self.mosaic_prob = self.cgf.mosaic_prob
        self.mixup_prob = self.cfg.mixup_prob

        ##***************************************************************##
        ## Data Loading  Params (like Data_path , etc )                  ##
        ##***************************************************************##

        assert os.path.isfile(self.json_file) ,"Cannot find {}".format(self.json_file)
        print("Load Dataset")
        self.coco_dataset = COCO(self.json_file)
        self.image_ids = self.coco_dataset.getImgIds()
        self.number_of_samples = len(self.image_ids)
        print("Number of Images {} ".format(self.number_of_samples))
        self.classes_inds = sorted(self.coco_dataset.getCatIds())

        cats = self.coco_dataset.loadCats(self.coco_dataset.getCatIds())
        self.classes_names = [c['name'] for c in cats]
        self.annotations = self._load_coco_annotations()
        self.samples_shapes = [self.img_size for _ in range(self.num_samples)]


        print("classes index : ",self.classes_inds)
        print("classes name in Dataset :",self.classes_names)


    def __len__(self):
        return self.number_of_samples


    def multi_shape(self):
        size_factor = self.img_size[1]* (1./self.img_size[0])
        multi_shapes = []
        # (14, 26)  # None; multi-size train: from 448(14*32) to 832(26*32), set None to disable it
        for size in list(range(*self.random_size)):
            random_input_h = int(32*size)
            random_input_w = 32*int(size*size_factor)
            multi_shapes.append([random_input_h,random_input_w])
        print("multi size training {}".format(multi_shapes))
        if self.logger:
            self.logger.write("Multi size training :{}\n".format(multi_shapes))

        iter_num = int(np.ceil(self.number_of_samples / self.batch_size))
        samples_shapes = []
        rand_idx  = len(multi_shapes)-1
        for it in range(iter_num):
            if it !=0 and it%10 ==0 :
                rand_idx = np.random.choice(list(range(len(multi_shapes))))
            for _ in  range(self.batch_size):
                samples_shapes.append(multi_shapes[rand_idx])
        return samples_shapes


    def shuffle(self):
        np.random.shuffle(self.annotations)
        print("Shuffle images list in {}".format(self.json_file))
        if self.logger:
            self.logger.write("shuffle {} images list ..\n".format(self.json_file))
        if self.random_size is not None:
            self.samples_shapes = self.multi_shape()

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes.keys():
            one_image_res = all_bboxes[image_id]
            for res in one_image_res:
                cls , conf , bbox = res[0] ,res[1],res[2]
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    'category_id': self.class_ids[self.classes.index(cls)],
                    'image_id': int(image_id),
                    'score': float(conf)})
        return detections

    def run_coco_eval(self, results, save_dir):
        convert_into_coco_format = self.convert_eval_format(results)
        file_locations = open('{}/results.json'.format(save_dir) , 'w')
        json.dump(convert_into_coco_format , file_locations)

        coco_det = self.coco_dataset.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco_dataset ,'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()

        str_result = redirect_string.getvalue()
        print(str_result)

        ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large = coco_eval.stats[:6]
        return ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large , str_result



    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.image_ids]

    def load_anno_from_ids(self, id_):
        pass

    def pull_item(self, index):
        pass

    def close_random_size(self):
        pass

    def __getitem__(self, idx):
        pass

    def mixup(self, origin_img, origin_labels, input_dim):
        pass
























