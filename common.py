python
import torch.nn as nn

class CST(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = SwinTransformerBlock(c_, c_, c_//32,1)

    def forward(self, x):
        return self.cv3(torch.cat((self.cv1(x), self.m(self.cv2(x))), dim=1))
        ......
