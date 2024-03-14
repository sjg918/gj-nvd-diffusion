# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6
    cfg.MODEL.DiffusionDet.USE_TIME_EMBED = True

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.STEP_WEIGHT = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])

    # Input YCrCb
    cfg.INPUT.USE_Gray_Proposals = False
    cfg.INPUT.USE_CbCr_Proposals = True
    cfg.INPUT.CbCr_thr = 20
    cfg.INPUT.sobel_mask_size = 3
    cfg.INPUT.Train_CbCr_mask_size = 0
    cfg.INPUT.Test_CbCr_mask_size = 15
    cfg.INPUT.PARK_W = 1240
    cfg.INPUT.PARK_H = 550

    # Stage Diffusion
    cfg.MODEL.DiffusionDet.stage_diffusion = (False, False, False, False, False)
    cfg.MODEL.DiffusionDet.stage_DDIM = False
    cfg.MODEL.DiffusionDet.DDIM_alpha_next = 0.7027
    cfg.MODEL.DiffusionDet.DDIM_c = 3.5496e-05
    cfg.MODEL.DiffusionDet.DDIM_sigma = 0.7114
    cfg.MODEL.DiffusionDet.diffusion_num = (50, 50, 50, 50, 50)
    cfg.MODEL.DiffusionDet.diffusion_num_scale = (5, 5, 5, 5, 5)
    cfg.MODEL.DiffusionDet.drop_high = True
    cfg.MODEL.DiffusionDet.diffusion_middle = False
    cfg.MODEL.DiffusionDet.diffusion_xy_scale = 1.0
    cfg.MODEL.DiffusionDet.diffusion_wh_scale = 1.0
