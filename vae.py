import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from  torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from logger import Logger

# torch.manual_seed(1)
e = 2.71828182
Batch_Size = 50
Num_Label = 10
Z_Dim = 50
Learning_Rate = 0.0005
Epoch = 30
Use_Cuda = torch.cuda.is_available()

Should_Download = False
tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ])
])
train_data = datasets.CIFAR10(
    root='./cifar10/',
    train=True,
    transform=tfs,
    download=Should_Download,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_Size, shuffle=True)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 3*32*32 => 16*32*32
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16*32*32 => 16*16*16
            nn.MaxPool2d(2),
            # 16*16*16 => 8*16*16
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # 8*16*16 => 8*8*8
            nn.MaxPool2d(2)
        )
        self.mean = nn.Linear(8*8*8, Z_Dim)
        self.var = nn.Linear(8*8*8, Z_Dim)
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.mean(x)
        var = self.var(x)
        # noise = torch.randn(var.size()).clamp(-0.5, 0.5) * 0.1
        noise = torch.randn(var.size())
        noise = Variable(noise, requires_grad=False)
        codes = mean + var.exp().mul(noise)
        return mean, var, codes

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rebuilt = nn.Linear(Z_Dim, 8*8*8)
        self.decoder = nn.Sequential(
            # 8 8 8 => 16 8 8
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16 8 8 => 16 16 16
            nn.ConvTranspose2d(16, 16, 2, stride=2),

            # 16 16 16 => 3 16 16
            nn.Conv2d(16, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            # 3 16 16 => 3 32 32
            nn.ConvTranspose2d(10, 10, 2, stride=2),
            nn.Conv2d(10, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
    def forward(self, x):
        rebuilt = self.rebuilt(x)
        rebuilt = rebuilt.view(-1,8,8,8)
        return self.decoder(rebuilt)

class RegularLoss(nn.Module):
    def __init__(self, num_label, z_dim):
        super(RegularLoss, self).__init__()
        self.num_label = num_label
        self.z_dim = z_dim
        self.center_table = Variable(torch.ones(num_label, z_dim)*10, requires_grad=False)

    def forward(self, mean, var, codes, labels):
        # 对隐含层进行聚类

        # # 衰减强度
        # decay = e
        # # 更新聚类中心
        # diverge = None
        # for label in range(self.num_label):
        #     indices = labels.data.eq(label).nonzero()
        #     if indices.size() == ():
        #         continue
        #     indices = indices.squeeze(1)
        #     # point_temp = torch.index_select(codes.data, 0, indices)
        #     # center_temp = point_temp.mean(0)
        #     # self.center_table.data[label] = (self.center_table.data[label] + (decay-1)*center_temp) / decay
        #     mean_temp = torch.index_select(mean, 0, Variable(indices, requires_grad=False))
        #     if diverge is None:
        #         diverge = mean_temp - self.center_table[label]
        #     else:
        #         diverge = torch.cat((diverge, mean_temp - self.center_table[label]), 0)
        # # 二阶正则惩罚
        # l2_loss = diverge * diverge

        l2_loss = mean * mean
        var_loss = var.exp() - (1 + var)

        return torch.mean(var_loss + l2_loss)
encoder = Encoder()
decoder = Decoder()

if Use_Cuda:
    encoder.cuda()
    decoder.cuda()

logger = Logger('./logs')
optimizer = torch.optim.Adam([{
        'params': encoder.parameters()
    }, {
        'params': decoder.parameters()
    }], lr=Learning_Rate)

scheduler = MultiStepLR(optimizer, milestones=[30,80,150,250], gamma=0.6)
decoder_loss = nn.MSELoss()
regulas_loss = RegularLoss(Num_Label, Z_Dim)

rel_step = 0
for epoch in range(Epoch):
    scheduler.step()
    for step, (pic, label) in enumerate(train_loader):
        rel_step += 1
        if Use_Cuda:
            x = Variable(pic).cuda()
            y = Variable(label).cuda()
        else:
            x = Variable(pic)
            y = Variable(label)
        mean, var, codes = encoder(x)
        de = decoder(codes)

        loss1 = decoder_loss(de, x)
        loss2 = regulas_loss(mean, var, codes, y)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(epoch, '-', rel_step,'| %.4f' % loss.data[0], '| de: %.4f' % loss1.data[0], '|rgl: %.4f' % loss2.data[0], '|%.4f' % torch.mean(codes.data))

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'loss': loss.data[0], # scalar
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, rel_step)

            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in encoder.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.numpy(), rel_step) # from Parameter to np.array

            for tag, value in decoder.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.numpy(), rel_step)
            # (3) Log the images
            info = {
                'decoder': de[:2].data.numpy(),
                'origin': pic[:2].numpy()
            }

            for tag, images in info.items():
                logger.image_summary(tag, images, rel_step)

print("End")
torch.save(encoder.state_dict(), './vae.pkl')


