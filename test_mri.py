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
parser.add_argument('--epoch_num', type=int, default=0, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=12, help='phase number of Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')


args = parser.parse_args()
epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list


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




optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/MRI_CS_unrolling_cs_0.33_1s_wonoise"% (args.model_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if epoch_num > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, epoch_num)))

nrtest = 21

PSNR_All = np.zeros([1, nrtest], dtype=np.float32)
SSIM_All = np.zeros([1, nrtest], dtype=np.float32)

Init_PSNR_All = np.zeros([1, nrtest], dtype=np.float32)
Init_SSIM_All = np.zeros([1, nrtest], dtype=np.float32)
with torch.no_grad():
    for img_no in range(nrtest):
        img_gt_np = Testing_labels[img_no, :]
        image_gt = torch.from_numpy(img_gt_np)
        batch_x = image_gt.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(1, 1, batch_x.shape[0], batch_x.shape[1])
        batch_y = batch_x
        PhiTb = FFT_Mask_ForBack()(batch_y, mask)
        net_input = PhiTb


        X_out = model(net_input, mask)

        initial_result = PhiTb.cpu().data.numpy().reshape(batch_x.shape[2], batch_x.shape[3])
        Prediction_value = (X_out.cpu().data.numpy().reshape(batch_x.shape[2], batch_x.shape[3]))
        X_init = np.clip(initial_result, 0, 1).astype(np.float64)
        X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
        init_PSNR = psnr(X_init * 255, (img_gt_np * 255).astype(np.float64))
        init_SSIM = ssim(X_init * 255, (img_gt_np * 255).astype(np.float64), data_range=255)
        rec_PSNR = psnr(X_rec * 255., (img_gt_np * 255).astype(np.float64))
        rec_SSIM = ssim(X_rec * 255., (img_gt_np * 255).astype(np.float64), data_range=255)

        im_rec_rgb = np.clip(X_rec * 255, 0, 255).astype(np.uint8)
        im_int_rgb = np.clip(X_init * 255, 0, 255).astype(np.uint8)


        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

        Init_PSNR_All[0, img_no] = init_PSNR
        Init_SSIM_All[0, img_no] = init_SSIM

print('\n')

init_data = "MRI: Avg Initial  PSNR/SSIM for %s is %.2f/%.4f" % (
    'test', np.mean(Init_PSNR_All), np.mean(Init_SSIM_All))
output_data = "MRI: Avg Proposed PSNR/SSIM for %s is %.2f/%.4f" % (
    'test', np.mean(PSNR_All), np.mean(SSIM_All))
print(init_data)
print(output_data)
