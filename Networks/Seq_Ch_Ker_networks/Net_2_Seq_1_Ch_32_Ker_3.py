# _2
# 변수 이름 정리
# normalize 를 미리 하는 것이 아닌 데이터를 입력할때마다 적용

# _6
# _2 구조에서 layers 형식으로 bn 추가

# _6_1
# accuracy 저장 추가
# training 코드는 따로 만듬

import torch
import torch.nn as nn

torch.manual_seed(289343610626200)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        channel_size = 32
        kernel_size = 3
        padding = 1
        self.layers1 = torch.nn.Sequential(nn.Conv2d(4, channel_size, kernel_size=kernel_size, stride=1, padding=padding), nn.BatchNorm2d(channel_size),
                                          nn.ReLU(),
                                          nn.Conv2d(channel_size, channel_size, kernel_size=kernel_size, stride=1, padding=padding), nn.BatchNorm2d(channel_size),
                                          nn.ReLU(),
                                          nn.Conv2d(channel_size, channel_size, kernel_size=kernel_size, stride=1, padding=padding), nn.BatchNorm2d(channel_size),
                                          nn.ReLU()
                                          )

        self.layers2 = torch.nn.Sequential(nn.Conv2d(channel_size*1, 1024, kernel_size=kernel_size, stride=1, padding=padding), nn.BatchNorm2d(1024),
                                           nn.ReLU())

        self.layers3 = torch.nn.Sequential(nn.Conv2d(1024 + channel_size*0, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                                           nn.Dropout2d(0.5),
                                           nn.Conv2d(512, 1, 3, 1, 1), nn.BatchNorm2d(1), nn.ReLU())

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()

        out1 = self.layers1(x)

        out_cat1 = out1
        out_cat2 = self.layers2(out_cat1)
        out_cat3 = out_cat2

        out_MLP = self.layers3(out_cat3)

        x = out_MLP[0, 0, 0, :]
        # x = x
        output = torch.div(
            torch.sub(
                x,
                torch.min(x)
            ),
            torch.sub(
                torch.max(x),
                torch.min(x)
            )
        )
        # output = x

        # print(out1.size())
        # print(out2.size())
        # print(out3.size())
        # print(out4.size())
        # print('\n')
        # print(out_cat1.size())
        # print(out_cat2.size())
        # print(out_cat3.size())
        # print('\n')
        # print(out_MLP.size())
        # print('\n')
        # print(x.size())
        # print(output.size())

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



