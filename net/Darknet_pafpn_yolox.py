import torch
from torch import nn
from net.backbone.draknet import CSPDarknet
from net.neck.pa_fpn import YOLO_PA_FPN
from net.head.yolox_head import YOLOX_Head
from net.loss.yolox_loss import YOLOXLoss
from utils.model_utils import load_model
from utils.util import sync_time
from cfg.config import opt
import time , numpy as np
from net.post_process import yolox_post_process
from data_loader.data_augment import preproc


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



# https://github.com/ultralytics/yolov5/blob/4fb8cb353f7d1945e3e1b270980c883c82297d2f/utils/torch_utils.py
def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                              padding=conv.padding, bias=True).to(conv.weight.device)
        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fusedconv


# https://github.com/ultralytics/yolov5/blob/4fb8cb353f7d1945e3e1b270980c883c82297d2f/models/yolo.py#L160-L169
def fuse_model(model):
    from net.backbone.esnet import ConvBNLayer
    for m in model.modules():
        if type(m) is ConvBNLayer and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


class Detector(object):
    def __init__(self, cfg):
        self.opt = cfg
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt.pretrained = None
        self.model = YoloBody_Nano(self.opt.num_classes , opt )
        print("Loading model {}".format(self.opt.load_model))
        self.model = load_model(self.model, self.opt.load_model)
        self.model.to(self.opt.device)
        self.model.eval()
        if "fuse" in self.opt and self.opt.fuse:
            print("==>> fuse model's conv and bn...")
            self.model = fuse_model(self.model)

    def run(self, images, vis_thresh, show_time=False):
        batch_img = True
        if np.ndim(images) == 3:
            images = [images]
            batch_img = False

        with torch.no_grad():
            if show_time:
                s1 = time.time()

            img_ratios, img_shape = [], []
            inp_imgs = np.zeros([len(images), 3, self.opt.test_size[0], self.opt.test_size[1]], dtype=np.float32)
            for b_i, image in enumerate(images):
                img_shape.append(image.shape[:2])
                img, r = preproc(image, self.opt.test_size, self.opt.rgb_means, self.opt.std)
                inp_imgs[b_i] = img
                img_ratios.append(r)

            if show_time:
                s2 = time.time()
                print("[pre_process] time {}".format(s2 - s1))

            inp_imgs = torch.from_numpy(inp_imgs).to(self.opt.device)
            yolo_outputs = self.model(inp_imgs, show_time=show_time)

            if show_time:
                s3 = sync_time(inp_imgs)
            predicts = yolox_post_process(yolo_outputs, self.opt.stride, self.opt.num_classes, vis_thresh,
                                          self.opt.nms_thresh, self.opt.label_name, img_ratios, img_shape)
            if show_time:
                s4 = sync_time(inp_imgs)
                print("[post_process] time {}".format(s4 - s3))
        if batch_img:
            return predicts
        else:
            return predicts[0]


if __name__ == '__main__':
    from thop import profile
    from cfg.config import opt

    image = torch.randn(1, 3, 320, 320)
    model_obj = YoloBody_Nano(opt.num_classes, opt)

    inputs = torch.rand(1, 3, 416, 416)
    total_ops, total_params = profile(model_obj, (inputs,))
    print("total_ops {}G, total_params {}M".format(total_ops / 1e9, total_params / 1e6))
    # total_ops 1.646935637G, total_params 2.150905M

    print("After \n")
    inputs = torch.rand(1, 3, 416, 416)
    model_obj.eval()
    model_obj = fuse_model(model_obj)
    total_ops, total_params = profile(model_obj, (inputs,))
    print("total_ops {} G, total_params {} M".format(total_ops / 1e9, total_params / 1e6))
    # total_ops 1.646935637G, total_params 2.150905M