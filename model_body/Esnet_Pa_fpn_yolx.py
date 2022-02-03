import torch
from torch import nn
from backbone.esnet import ESNet
from neck.pa_fpn import YOLO_PA_FPN
from head.yolox_head import YOLOX_Head


class YoloBody(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        depth, width = 0.24 , 1.0
        depthwise = True
        self.backbone = ESNet()
        self.in_channels = [96,192,384]
        self.neck = YOLO_PA_FPN(depth, width=1, in_channels=self.in_channels ,depthwise=depthwise)
        self.head = YOLOX_Head(num_classes, width=1, in_channels=self.in_channels ,depthwise=depthwise)



    def forward(self, x):
        backbone_output = self.backbone(x)
        # for x in backbone_output :
        #     print(x.shape)
        fpn_outs = self.neck(backbone_output)
        # print("\n\n")
        # for x in fpn_outs :
        #     print(x.shape)
        outputs = self.head.forward(fpn_outs)
        return outputs


if __name__ == '__main__':
    image = torch.randn(1, 3, 320, 320)
    model_obj = YoloBody(1)
    output = model_obj(image)
    for x in output:
        print(x.shape)
