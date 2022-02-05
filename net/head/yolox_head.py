import torch
import torch.nn as nn
from net.backbone.draknet import BaseConv, DWConv


class YOLOX_Head(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=None, act="silu", depthwise=False, ):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024]
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   P3_out : (80, 80, 256)
        #   P4_out : (40, 40, 512)
        #   P5_out : (20, 20, 1024)
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)

            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   (80, 80, num_classes)
            #   (40, 40, num_classes)
            #   (20, 20, num_classes)
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   reg_pred : (80, 80, 4)
            #   reg_pred : (40, 40, 4)
            #   reg_pred : (20, 20, 4)
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)

            # ---------------------------------------------------#
            #   obj_pred :(80, 80, 1)
            #   obj_pred :(40, 40, 1)
            #   obj_pred :(20, 20, 1)
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            # convert into yolooutput : [ W , H , (1 + 4 + number_of_class)]
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
