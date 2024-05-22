# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import numba
import torch
import cv2

from detectron2.data import detection_utils as utils
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
import pycocotools.mask as mask_util
from .data_augmentation import (
    RandomFlip,
    RandomCrop,
    ResizeShortestEdge,
    TransformList,
    apply_augmentations,
)


__all__ = ["DiffusionDetDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(RandomFlip())
    # ResizeShortestEdge
    tfm_gens.append(ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    
    return tfm_gens


@numba.jit(nopython=True, cache=True)
def get_NMS_Boxmask(Boxmask, W, H, size):
    for i in range(W):
        for j in range(H):
            if Boxmask[j, i] == False:
                continue

            for p in range(-(size // 2), size // 2 + 1):
                for q in range(-(size // 2), size // 2 + 1):
                    if j + q < 0 or j + q >= H or i + p < 0 or i + p >= W:
                        continue

                    if p == 0 and q == 0:
                        continue
                    
                    Boxmask[j + q, i + p] = False
                    continue
                continue

            continue
        continue
    return


class DiffusionDetDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DiffusionDet.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            # self.crop_gen = [
            #     T.ResizeShortestEdge(short_edge_length=(800,800), max_size=1333, sample_style="choice"),
            # ]
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        x = np.linspace(1, cfg.INPUT.PARK_W, cfg.INPUT.PARK_W)
        y = np.linspace(1, cfg.INPUT.PARK_H, cfg.INPUT.PARK_H)
        X, Y = np.meshgrid(x, y)
        grid = np.concatenate((X.reshape(cfg.INPUT.PARK_H, cfg.INPUT.PARK_W, 1), 
                            Y.reshape(cfg.INPUT.PARK_H, cfg.INPUT.PARK_W, 1)), axis=2)
        self.grid = grid.astype(np.int64)

        if is_train:
            self.cbcr_mask_size = cfg.INPUT.Train_CbCr_mask_size
        else:
            self.cbcr_mask_size = cfg.INPUT.Test_CbCr_mask_size
        self.gray_proposals = cfg.INPUT.USE_Gray_Proposals
        self.edge_thr = cfg.INPUT.CbCr_thr
        self.sobel_mask_size = cfg.INPUT.sobel_mask_size
        self.input_park_w = cfg.INPUT.PARK_W
        self.input_park_h = cfg.INPUT.PARK_H

    def get_CrCbmap(self, img):

        if self.gray_proposals:
            Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Gray_x = np.abs(cv2.Sobel(Gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=self.sobel_mask_size))
            Gray_y = np.abs(cv2.Sobel(Gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=self.sobel_mask_size))
            GrayEdge = (Gray_x / 2 + Gray_y / 2).astype(np.uint8)
            Boxmask = GrayEdge > self.edge_thr
        else:
            YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            CrEdge_x = np.abs(cv2.Sobel(YCrCb[:, :, 1], ddepth=cv2.CV_64F, dx=1, dy=0, ksize=self.sobel_mask_size))
            CrEdge_y = np.abs(cv2.Sobel(YCrCb[:, :, 1], ddepth=cv2.CV_64F, dx=0, dy=1, ksize=self.sobel_mask_size))
            CrEdge = (CrEdge_x / 2 + CrEdge_y / 2).astype(np.uint8)
            CbEdge_x = np.abs(cv2.Sobel(YCrCb[:, :, 2], ddepth=cv2.CV_64F, dx=1, dy=0, ksize=self.sobel_mask_size))
            CbEdge_y = np.abs(cv2.Sobel(YCrCb[:, :, 2], ddepth=cv2.CV_64F, dx=0, dy=1, ksize=self.sobel_mask_size))
            CbEdge = (CbEdge_x / 2 + CbEdge_y / 2).astype(np.uint8)

            Crmask = CrEdge > self.edge_thr
            Cbmask = CbEdge > self.edge_thr
            Boxmask = np.logical_or(Crmask, Cbmask)
        
        if self.cbcr_mask_size > 0:
            get_NMS_Boxmask(Boxmask, self.input_park_w, self.input_park_h, self.cbcr_mask_size)

        vaild_points = self.grid[Boxmask]

        return vaild_points

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        vaild_points = self.get_CrCbmap(image)
        #image = np.concatenate((image, CrEdge.reshape(550, 1240, 1), CbEdge.reshape(550, 1240, 1)), axis=2)

        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = apply_augmentations(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = apply_augmentations(self.tfm_gens, image)
            else:
                image, transforms = apply_augmentations(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w
        vaild_points = vaild_points.astype(np.float32)
        for trf in transforms:
            vaild_points = trf.apply_coords(vaild_points)
            continue
        mask1 = np.logical_and(vaild_points[:, 1] < image_shape[0], vaild_points[:, 1] > 0)
        mask2 = np.logical_and(vaild_points[:, 0] < image_shape[1], vaild_points[:, 0] > 0)
        mask = np.logical_and(mask1, mask2)
        vaild_points = vaild_points[mask]
        
        vaild_points[:, 0] = vaild_points[:, 0] / image_shape[1]
        vaild_points[:, 1] = vaild_points[:, 1] / image_shape[0]
        vaild_points = (vaild_points * 2 - 1) * 2

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["vaild"] = torch.from_numpy(np.ascontiguousarray(vaild_points))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
