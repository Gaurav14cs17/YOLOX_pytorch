import torch
from torch import nn
from net.backbone.draknet import CSPDarknet
from net.neck.pa_fpn import YOLO_PA_FPN
from net.head.yolox_head import YOLOX_Head
from net.loss.yolox_loss import YOLOXLoss
from utils.model_utils import load_model
from utils.util import sync_time


class YoloBody_Nano(nn.Module):
    def __init__(self, num_classes, opt, phi='nano'):
        super().__init__()
        self.opt = opt
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.out_indices = ("dark3", "dark4", "dark5")
        self.in_channel = [256, 512, 1024]

        self.backbone = CSPDarknet(dep_mul=depth, wid_mul=width, out_features=self.out_indices, depthwise=opt.depth_wise)
        self.neck = YOLO_PA_FPN(depth, width, in_channels = self.in_channel ,depthwise=depthwise)

        self.head = YOLOX_Head(num_classes, width=width, in_channels = self.in_channel ,depthwise=depthwise)

        self.loss = YOLOXLoss(opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums, strides=opt.stride,
                              in_channels=self.in_channel)

        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def backbone_output(self, out_features):
        return [out_features[f] for f in self.out_indices]

    def forward(self, x, targets=None, show_time=False):
        with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
            if show_time:
                s1 = sync_time(x)
            backbone_output = self.backbone(x)
            neck_input = self.backbone_output(backbone_output)


            fpn_outs = self.neck(neck_input)


            yolo_outputs = self.head.forward(fpn_outs)
            if show_time:
                s2 = sync_time(x)
                print("[inference] batch={} time: {}s".format("x".join([str(i) for i in x.shape]), s2 - s1))

            if targets is not None:
                loss = self.loss(yolo_outputs, targets)

            if targets is not None:
                loss = self.loss(yolo_outputs, targets)

            if targets is not None:
                return yolo_outputs, loss
            else:
                return yolo_outputs


if __name__ == '__main__':
    from cfg.config import opt

    image = torch.randn(1, 3, 640, 640)
    model_obj = YoloBody_Nano(1, opt)
    output = model_obj(image)
    for x in output:
        print(x.shape)
