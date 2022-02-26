from model_utils.model import convnext_tiny ,convnext_small
import torch
import os
import argparse
import numpy as np

torch.set_grad_enabled(False)

DATASET_TO_CLASSES = {
    "imagenet-1k": 1000,
    "imagenet-21k": 21841,
}
MODEL_TO_METHOD = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
}
TF_MODEL_ROOT = "saved_models"


def parse_args():
    parser = argparse.ArgumentParser(description="Conversion of the PyTorch pre-trained ConvNeXt weights to TensorFlow.")
    parser.add_argument("-d","--dataset",default="imagenet-1k",type=str,required=False,choices=["imagenet-1k", "imagenet-21k"],help="Name of the pretraining dataset.",)
    parser.add_argument(
        "-m",
        "--model-name",
        default="convnext_tiny",
        type=str,
        required=False,
        choices=[
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
            "convnext_xlarge",
        ],
        help="Name of the ConvNeXt model variant.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=224,
        type=int,
        required=False,
        choices=[224, 384],
        help="Image resolution.",
    )
    parser.add_argument("-c","--checkpoint-path",default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",type=str,
        required=False,
        help="URL of the checkpoint to be loaded.",
    )
    return vars(parser.parse_args())


def main(args):
    print(f'Model: {args["model_name"]}')
    print(f'Image resolution: {args["resolution"]}')
    print(f'Dataset: {args["dataset"]}')
    print(f'Checkpoint URL: {args["checkpoint_path"]}')

    print("Instantiating PyTorch model and populating weights...")
    model_method = MODEL_TO_METHOD[args["model_name"]]
    convnext_model_pt = model_method(args["checkpoint_path"], num_classes=DATASET_TO_CLASSES[args["dataset"]])
    convnext_model_pt.eval()

    print("Instantiating TensorFlow model...")

    if "22k_1k" not in args["checkpoint_path"]:
        model_name = (
            f'{args["model_name"]}_1k'
            if args["dataset"] == "imagenet-1k"
            else f'{args["model_name"]}_21k'
        )
    else:
        model_name = f'{args["model_name"]}_21k_1k'



if __name__ == "__main__":
    args = parse_args()
    main(args)