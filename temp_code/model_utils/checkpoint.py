import os
import shutil
from loguru import logger
import torch


def save_checkpoint(state , is_best , save_dir , model_name = ""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir ,model_name + "_ckpt.pth")
    torch.save(state , filename)
    if is_best:
        best_filename = os.path.join(save_dir , "best_ckpt.pth")
        shutil.copyfile(filename , best_filename)


def load_ckpt(model , ckpt ):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model ,v in model_state_dict.item():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt . please double click and see if this is desired".format(key_model)
            )
            continue
        value_of_ckpt = ckpt[key_model]
        if v.shape != value_of_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {} while shape of {} in model is {}".format(key_model ,value_of_ckpt.shape ,key_model , v.shape)
            )
            continue

    model.load_state_dict(load_dict , strict=False)
    return model



