import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from torch.utils.data import DataLoader
from generator import *
from discriminator import *
from scanDataset import *

def create_coordsZXY(start_idx):
    z = start_idx + 8
    x = np.random.randint(90, 411)
    y = np.random.randint(90, 411)
    return np.array([z, y, x])

def compute_mse_loss(real, protected_scan, start_idx, end_idx, length, action="inject", save_img_result=False):
    if start_idx > 32 and (start_idx + 32) < length:
        try:
            protected_scan_np = protected_scan.clone().permute(1,0,2,3).squeeze().detach().cpu().numpy() 
            netManipulator.load_target_scan(data[1][0])
            protected_scan_np = (protected_scan_np - protected_scan_np.min()) / (protected_scan_np.max() - protected_scan_np.min()) * (netManipulator.scan[start_idx:end_idx].max() - netManipulator.scan[start_idx:end_idx].min()) + netManipulator.scan[start_idx:end_idx].min()
            netManipulator.scan[start_idx:end_idx] = protected_scan_np
            coords = create_coordsZXY(start_idx)
            netManipulator.tamper(coords, action=action, isVox=True)
            target = torch.tensor(netManipulator.scan[start_idx:end_idx]).unsqueeze(1).double().to(device)
            target = (target - target.min()) / (target.max() - target.min())
            target = 2 * target - 1
            if save_img_result:
                all_imgs = torch.cat((real, protected_scan, target),0)
                vutils.save_image(vutils.make_grid(all_imgs, padding=5, normalize=True), "mse_"+str(iters)+".png") 
            mse_loss_value = mse_loss(protected_scan, target) * -1
            return mse_loss_value
        except:
            return 0
    else:
        return 0

#SETUP
image_size = 512 #must be this size
ngpu = 1
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#DATASET
data_directory = "MITS-GAN/data/"
train_dataset = ScanDataset(data_directory)
batch_size = 1
sub_batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#INIT NETWORK PARAMS
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#GENERATOR
netG = Generator(1,1,512).double().to(device)
netG.apply(weights_init)
print(netG)

#DISCRIMINATOR
netD = Discriminator(ngpu).double().to(device)
netD.apply(weights_init)
print(netD)

#CT-GAN NETWORK
from procedures.attack_pipeline import *
netManipulator = scan_manipulator()

#TRAINING SETUP
criterion = nn.BCELoss()
mse_loss = nn.MSELoss()
iters = 0
num_epochs = 5
real_label = 1.
protected_label = 0.
lr_g = 0.0002
lr_d = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        batch = data[0].permute(1,0,2,3).double()
        num_sub_batches = batch.size(0) // sub_batch_size
        for j in range(num_sub_batches):
            start_idx = j * sub_batch_size
            end_idx = (j + 1) * sub_batch_size
            sub_batch = batch[start_idx:end_idx]
            netD.zero_grad()
            real = sub_batch.to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.double, device=device)
            output = netD(real).view(-1)
        
            errD_real = criterion(output, label)
            errD_real.backward()

            protected = netG(real)
            label.fill_(protected_label)
            output = netD(protected.detach()).view(-1)
            errD_protected = criterion(output, label)
            errD_protected.backward()
            errD = errD_real + errD_protected
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(protected).view(-1)
            errG = criterion(output, label)
            mse_loss_value = compute_mse_loss(real, protected, start_idx, end_idx, batch.size(0)) if j % 2 == 0 else compute_mse_loss(real ,protected, start_idx, end_idx, batch.size(0), "remove")
            errG = errG + 0.1 * mse_loss_value
            errG.backward()
            optimizerG.step()
        
        iters += 1 
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t' % (epoch, num_epochs, i, len(train_loader), errD.item(), errG.item()))
