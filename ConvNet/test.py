from backbone import Model
from neck import FPN
import torch



back = Model()
neck_obj = FPN()


image = torch.randn((1, 3, 640, 640))
back_out = neck_obj(back(image))
for x in back_out:
    print(x.shape)

