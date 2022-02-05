import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_body.Esnet_Pa_fpn_yolx import YoloBody
from model_body.yolo_training import YOLOLOSS, weights_init
from utils.callbacks import LossHistory
from data_loader.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    classes_path = 'model_data/voc_classes.txt'
    model_path = 'logs/ep038-loss7.294-val_loss7.180.pth'
    input_shape = [320, 320]
    phi = 'nano'
    mosaic = False
    Cosine_scheduler = False

    Freeze_Train = False
    UnFreeze_Train = True





    num_workers = 1
    train_annotation_path = 'model_data/2012_train.txt'
    val_annotation_path = 'model_data/2012_val.txt'

    class_names, num_classes = get_classes(classes_path)
    model = YoloBody(num_classes)
    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # torch.Size([32, 25, 40, 40])
    # torch.Size([32, 25, 20, 20])
    # torch.Size([32, 25, 10, 10])

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()


    strides = [8,16 , 32  ] # [320/40 , 320/20 ,320/10]
    yolo_loss = YOLOLOSS(num_classes , strides=strides)
    loss_history = LossHistory("logs/")
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)


    if Freeze_Train:
        Init_Epoch = 0
        Freeze_Epoch = 50
        Freeze_batch_size = 8
        Freeze_lr = 1e-3
        print("start transfer learning")

        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("error")
        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_scheduler:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()




    if UnFreeze_Train:
        UnFreeze_Epoch = 100
        Unfreeze_batch_size = 16
        Unfreeze_lr = 1e-4

        print("model train from skreach")
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = 0
        end_epoch = UnFreeze_Epoch
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("error")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_scheduler:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,train=False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)

        # if UnFreeze_Train:
        #     for param in model.backbone.parameters():
        #         param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
