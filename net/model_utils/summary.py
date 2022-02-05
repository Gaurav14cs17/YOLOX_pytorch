import torch
from torchsummary import summary
from model_body.yolox import YoloBody

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = YoloBody(80, 'nano').to(device)
    print(m)
    summary(m, input_size=(3, 320, 320))