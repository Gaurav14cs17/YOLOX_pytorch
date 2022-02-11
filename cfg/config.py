import os
import sys
from easydict import EasyDict
from utils.util import merge_opt


def update_nano_tiny(cfg, inp_params):
    # yolo-nano, yolo-tiny config:
    cfg.scale = cfg.scale if 'scale' in inp_params else (0.1, 0.5)
    cfg.test_size = cfg.test_size if 'test_size' in inp_params else (320, 320)
    cfg.enable_mixup = cfg.enable_mixup if 'enable_mixup' in inp_params else False
    cfg.mosaic_prob = cfg.mosaic_prob if 'mosaic_prob' in inp_params else 0.5
    if 'random_size' not in inp_params:
        if cfg.random_size is not None:
            cfg.random_size = (5, 10)
    if 'nano' in cfg.backbone:
        cfg.depth_wise = True
    return cfg


opt = EasyDict()
opt.exp_id = "coco_CSPDarknet-s_640x640"
opt.dataset_path = "./model_data/"
opt.images_dataset_path = "/content/drive/MyDrive/"
# opt.dataset_path = r"D:\work\public_dataset\coco2017"  # Windows system
opt.backbone = "CSPDarknet-nano"  # CSPDarknet-nano, CSPDarknet-tiny, CSPDarknet-s, CSPDarknet-m, l, x
opt.input_size = (320, 320)
opt.random_size = (14, 26)  # None; multi-size train: from 448(14*32) to 832(26*32), set None to disable it
opt.test_size = (320, 320)  # evaluate size
opt.gpus = "0"  # "-1" "0" "3,4,5" "0,1,2,3,4,5,6,7" # -1 for cpu
opt.batch_size = 100
opt.master_batch_size = -1  # batch size in first gpu. -1 means: master_batch_size=batch_size//len(gpus)
opt.num_epochs = 300

# coco 80 classes
opt.label_name = ['face']

# TODO: support MOT(multi-object tracking) like FairMot/JDE when reid_dim > 0
opt.reid_dim = 0  # 128  used in MOT, will add embedding branch if reid_dim>0
# opt.label_name = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
#                   'motor']
# tracking id number of label_name in MOT train dataset
opt.tracking_id_nums = None  # [1829, 853, 323, 3017, 295, 159, 215, 79, 55, 749]

# base params
opt.warmup_lr = 0  # start lr when warmup
opt.basic_lr_per_img = 0.01 / 64.0
opt.scheduler = "yoloxwarmcos"
opt.no_aug_epochs = 15  # close mixup and mosaic augments in the last 15 epochs
opt.min_lr_ratio = 0.05
opt.weight_decay = 5e-4
opt.warmup_epochs = 5
opt.depth_wise = False  # depth_wise conv is used in 'CSPDarknet-nano'
opt.stride = [8, 16, 32]  # YOLOX down sample ratio: 8, 16, 32

# train augments
opt.degrees = 10.0  # rotate angle
opt.translate = 0.1
opt.scale = (0.1, 2)
opt.shear = 2.0
opt.perspective = 0.0
opt.enable_mixup = True
opt.seed = None  # 0
opt.mosaic_prob = 1.
opt.mixup_prob = 1.
opt.data_num_workers = 1

opt.momentum = 0.9
opt.vis_thresh = 0.3  # inference confidence, used in 'predict.py'
opt.load_model = 'D:\labs\object_detection_model\YOLOX_pytorch\cfg\exp\coco_CSPDarknet-s_640x640\model_last.pth'
opt.ema = True  # False, Exponential Moving Average
opt.grad_clip = dict(max_norm=35, norm_type=2)  # None, clip gradient makes training more stable
opt.print_iter = 1  # print loss every 1 iteration
opt.val_intervals = 2  # evaluate val dataset and save best ckpt every 2 epoch
opt.save_epoch = 1  # save check point every 1 epoch
opt.resume = True  # resume from 'model_last.pth' when set True
opt.use_amp = False  # True, Automatic mixed precision
opt.cuda_benchmark = True
opt.nms_thresh = 0.65  # nms IOU threshold in post process
opt.occupy_mem = False  # pre-allocate gpu memory for training to avoid memory Fragmentation.

opt.rgb_means = [0.485, 0.456, 0.406]
opt.std = [0.229, 0.224, 0.225]

opt, input_params = merge_opt(opt, sys.argv[1:])
if opt.backbone.lower().split("-")[1] in ["tiny", "nano"]:
    opt = update_nano_tiny(opt, input_params)

# do not modify the following params
opt.train_ann = opt.dataset_path + "/annotations/instances_train2017.json"
opt.val_ann = opt.dataset_path + "/annotations/instances_val2017.json"
opt.data_dir = opt.dataset_path
if isinstance(opt.label_name, str):
    new_label = opt.label_name.split(",")
    print('[INFO] change param: {} {} -> {}'.format("label_name", opt.label_name, new_label))
    opt.label_name = new_label
opt.num_classes = len(opt.label_name)
opt.gpus_str = opt.gpus
opt.gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
if opt.master_batch_size == -1:
    opt.master_batch_size = opt.batch_size // len(opt.gpus)
rest_batch_size = opt.batch_size - opt.master_batch_size
opt.chunk_sizes = [opt.master_batch_size]
for i in range(len(opt.gpus) - 1):
    slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
    if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
    opt.chunk_sizes.append(slave_chunk_size)
opt.root_dir = os.path.dirname(__file__)
opt.save_dir = os.path.join(opt.root_dir, 'exp', opt.exp_id)
opt.video_dir = "./"
if opt.resume and opt.load_model == '':
    opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
if opt.random_size is not None and (opt.random_size[1] - opt.random_size[0] > 1):
    opt.cuda_benchmark = False
if opt.random_size is None:
    opt.test_size = opt.input_size
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
print("\n{} final config: {}\n{}".format("-" * 20, "-" * 20, opt))
