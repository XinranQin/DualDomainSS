import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from skimage.measure import compare_ssim as ssim
import math
parser = ArgumentParser(description='Unroll-Net')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=12, help='phase number of Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--batch_size', type=int, default='4', help='batch size')


args = parser.parse_args()
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list
batch_size = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Phi_data_Name = './%s/mask1s0.25.mat' % (args.matrix_dir)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['k']
mask_matrix_np = np.fft.fftshift(mask_matrix)

mask_matrix = torch.from_numpy(mask_matrix_np).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)

nrtrain = 300
Training_data_Name = 'MRI_data.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['MRI_data']
Testing_labels = Training_labels.transpose(2,0,1)[nrtrain:,:]
Training_labels = Training_labels.transpose(2,0,1)[:nrtrain,:]


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        x_dim_0 = x.shape[0]
        x_dim_1 = x.shape[1]
        x_dim_2 = x.shape[2]
        x_dim_3 = x.shape[3]
        x = x.view(-1, x_dim_2, x_dim_3, 1)
        y = torch.zeros_like(x)
        z = torch.cat([x, y], 3)
        fftz = torch.fft(z, 2)
        z_hat = torch.ifft(fftz * mask, 2)
        x = z_hat[:, :, :, 0:1]
        x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
        return x


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 1, 3, 3)))

        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 16, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias = nn.Parameter(torch.full([32], 0.01))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv5 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv6 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv7 = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 16, 3, 3)))


    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x


        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x_forward = F.conv2d(x_forward, self.conv3, bias=self.bias, padding=1)
        x = F.relu(x_forward)
        x = F.conv2d(x, self.conv4, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv5, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv6, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv7, padding=1)
        x_G = F.conv2d(x, self.conv_G, padding=1)
        x_out = x_input + x_G



        return x_out


# Define Network
class Network(torch.nn.Module):
    def __init__(self, LayerNo):
        super(Network, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()
        for i in range(LayerNo):
            onelayer.append(BasicBlock())
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        for i in range(self.LayerNo):
            x = self.fcs[i](x, self.fft_forback, PhiTb, mask)

        x_final = x

        return x_final

model = Network(layer_num)
model = nn.DataParallel(model)
model = model.to(device)



class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len



rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/MRI_CS_unrolling_cs_0.33_1s_wonoise"% (args.model_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir


Training_labels = torch.Tensor(Training_labels).float()

# Training loop
for epoch_i in range(start_epoch, end_epoch):
    for data in rand_loader:
        # gennerate cs image
        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
        PhiTb = FFT_Mask_ForBack()(batch_x, mask)
        gamma = (torch.FloatTensor(PhiTb.size()).normal_(mean=0, std=2 / 255).cuda())
        net_input = PhiTb + gamma
        x_output = model(net_input, mask)

        # Loss in measurement domian
        loss_range = torch.mean(torch.pow(FFT_Mask_ForBack()(x_output,mask) - (PhiTb-gamma), 2))
        z = x_output.detach()
        net_input1 = FFT_Mask_ForBack()(z, mask)
        with torch.no_grad():
            z_p = model(net_input1, mask)
        res = (z_p - z).detach()
        mask_s = torch.ones_like(res)
        mask_s = F.dropout(mask_s, 0.5) * 0.5
        mask_s = torch.where(mask_s == 0, mask_s - 1, mask_s)
        r = (mask_s * res)
        net_input1 = FFT_Mask_ForBack()(x_output+r, mask)
        x_output1 = model(net_input1, mask)
        gamma = torch.Tensor([0.1]).to(device)
        loss_image = torch.mean(torch.pow((x_output1 - (x_output - r)),2))
        loss_all = loss_range + gamma*loss_image
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        output_data = "[%02d/%02d] Total Loss: %.5f, Loss in image: %.5f\n" % (
            epoch_i, end_epoch, loss_all.item(), loss_image.item())
        print(output_data)

