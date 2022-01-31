import torch
from torch import nn
from models.backbone.draknet import CSPDarknet
from models.neck.pa_fpn import YOLO_PA_FPN
from models.head.yolox_head import YOLOX_Head


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi='s'):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.in_features = ("dark3", "dark4", "dark5")

        self.backbone = CSPDarknet(depth, width, act='silu')
        self.neck = YOLO_PA_FPN(depth, width, depthwise=depthwise)
        self.head = YOLOX_Head(num_classes, width, depthwise=depthwise)

    def backbone_output(self, out_features):
        return [out_features[f] for f in self.in_features]

    def forward(self, x):
        backbone_output = self.backbone(x)
        neck_input = self.backbone_output(backbone_output)

        fpn_outs = self.neck(neck_input)
        outputs = self.head.forward(fpn_outs)
        return outputs


if __name__ == '__main__':
    image = torch.randn(1, 3, 320, 320)
    model_obj = YoloBody(1)
    output = model_obj(image)
    for x in output:
        print(x.shape)
