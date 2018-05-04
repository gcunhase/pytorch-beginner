__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import numpy as np


"""
    Modified by Gwena on May 4th 2018 to recover the hidden state after the encoder
    Used for feature extraction
"""

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 2
saving_step = 10
batch_size = 64  # 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h2, h1


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output, hidden = model(img)
        loss = criterion(output, img)
        loss2 = criterion(model.decoder(hidden), img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, loss2:{:.4f}'.format(epoch+1, num_epochs, loss.data[0], loss2.data[0]))

    if epoch % saving_step == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))
        out_restored = model.decoder(hidden)
        pic_restored = to_img(out_restored.cpu().data)
        save_image(pic_restored, './dc_img/image_{}_restored.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
# Get weights
print("Model keys: {}".format(model.state_dict().keys()))
# W = model.state_dict()['fc1.weight']


# Test with loaded model and last train img
model = autoencoder().cuda()
model.load_state_dict(torch.load('./conv_autoencoder.pth'))

pic = to_img(img.cpu().data)
save_image(pic, './dc_img/img_in.png')
output, hidden = model(img)
pic_out = to_img(output.cpu().data)
save_image(pic_out, './dc_img/img_out.png')
out_restored = model.decoder(hidden)
pic_restored = to_img(out_restored.cpu().data)
save_image(pic_restored, './dc_img/img_restored.png')

