import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import math
from time import time
import cv2
from skimage.measure import compare_ssim as ssim
import glob

parser = ArgumentParser(description='Dual-Domain-newtwork')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

oju8parser = ArgumentParser(description='Unroll-Net-plus')
parser.add_argument('--epoch', type=int, default=390, help='epoch number of test')
parser.add_argument('--layer_num', type=int, default=12, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

args = parser.parse_args()

epoch = args.epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912  # number of training blocks
batch_size = 64


Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

Qinit = np.linalg.pinv(Phi_input)


def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size - np.mod(row, block_size)
    col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def col2im_CS_torch(X_col, row_new, col_new):
    block_size = 33
    r_b = row_new // 33
    c_b = col_new // 33
    X_col = X_col.view(r_b, c_b, block_size, block_size)
    X_col = X_col.permute(0, 2, 1, 3)
    X_rec = torch.reshape(X_col, (-1, col_new))
    X_rec = torch.reshape(X_rec, (row_new, col_new))

    return X_rec.unsqueeze(0).unsqueeze(1)


def img2col_torch(Ipad, block_size):
    [row, col] = Ipad.shape
    img_col = Ipad.view(-1, block_size, col)
    img_col = img_col.view(img_col.shape[0], block_size, -1, block_size)
    img_col = img_col.permute(0, 2, 1, 3)
    img_col = torch.reshape(img_col, (-1, 1, block_size, block_size))
    img_col = torch.reshape(img_col, (-1, block_size * block_size))

    return img_col

def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row / block_size
    col_block = col / block_size
    block_num = int(row_block * col_block)
    img_col = np.zeros([block_size ** 2, block_num])
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# Define Basic Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias = nn.Parameter(torch.full([32], 0.01))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias1 = nn.Parameter(torch.full([32], 0.01))
        self.conv5 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv6 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv7 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb

        x_input = x.view(-1, 1, 33, 33)
        # x_input = col2im_CS_torch(x_input, row_new, col_new)
        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x_forward = F.conv2d(x_forward, self.conv3, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x = F.conv2d(x_forward, self.conv4,bias=self.bias, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv5, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv6, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv7, padding=1)
        x_G = F.conv2d(x, self.conv_G, padding=1)
        x_pred = x_input + x_G
        x_pred = x_pred.view(-1, 33*33)
        #x_pred = img2col_torch(x_pred[0, 0], 33)

        return x_pred


# Define Network
class Network(torch.nn.Module):
    def __init__(self, LayerNo):
        super(Network, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, xinit=None):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        if xinit is None:
            x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        else:
            x = xinit

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTPhi, PhiTb)

        x_final = x

        return x_final


model = Network(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1  # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

model_dir = "./%s/DDSS_%d_ratio_%d_lr_%.4f_unsuper" % (args.model_dir, layer_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, epoch)))

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)


def together(inputs, S, H, L):
    inputs = torch.reshape(inputs, [-1, 33, 33])
    inputs = torch.cat(torch.split(inputs, split_size_or_sections=H * S, dim=0), dim=2)
    inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
    return inputs

mse = nn.MSELoss()
# Training loop

test_dir = os.path.join(args.data_dir, args.test_name)
filepaths = glob.glob(test_dir + '/*.tif')
ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
inner_loop = 200
for img_no in range(ImgNum):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*1)
    imgName = filepaths[img_no]
    Img = cv2.imread(imgName, 1)
    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Img_rec_yuv = Img_yuv.copy()
    Iorg_y = Img_yuv[:, :, 0]
    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
    Icol = img2col_py(Ipad, 33).transpose() / 255.0
    Img_output = Icol
    start = time()
    batch_x = torch.from_numpy(Img_output)
    batch_x = batch_x.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
    for i in range(inner_loop):
        gamma = (torch.FloatTensor(Phix.size()).normal_(mean=0, std=2 / 255).cuda())
        x_output = model(Phix + gamma, Phi, Qinit)
        ## loss in range domain
        loss_range = torch.mean(
            torch.pow(torch.mm(x_output, torch.transpose(Phi, 0, 1)) - (Phix), 2))
        ## sampling r
        z = x_output
        PhiNx = torch.mm(z, torch.transpose(Phi, 0, 1))
        z_prime = model(PhiNx, Phi, Qinit)
        res = (z_prime - z).view(-1, 1, 33, 33)
        mask = torch.ones_like(res)
        mask = F.dropout(mask, 0.5) * 0.5
        mask = torch.where(mask == 0, -1 * mask, mask)
        r = (mask * res.detach()).view(-1, 1089)
        PhiNx = torch.mm(x_output.detach() + r, torch.transpose(Phi, 0, 1))
        x_input1 = PhiNx
        x_output1 = model(x_input1, Phi, Qinit)
        # loss in image domian
        loss_image = torch.mean(torch.pow(x_output1 - (x_output - r), 2))
        loss_all = loss_range + 0.1 * (loss_image)
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
    with torch.no_grad():
        x_output = model(Phix, Phi, Qinit)
        end = time()
    Prediction_value = x_output.cpu().data.numpy()
    X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0, 1)
    rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
    rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)

    print("[%02d/%02d] Run time for %s , PSNR is %.2f, SSIM is %.4f" % (
        img_no, ImgNum, imgName, rec_PSNR, rec_SSIM))

    Img_rec_yuv[:, :, 0] = X_rec * 255

    im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

    resultName = imgName.replace(args.data_dir, args.result_dir)

    PSNR_All[0, img_no] = rec_PSNR
    SSIM_All[0, img_no] = rec_SSIM

print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (
    cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch)
print(output_data)




