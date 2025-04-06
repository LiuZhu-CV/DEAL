import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.utilities import pre_process, ssim
from models.model_ng import NG as ng
from models.model_ne import NE as ne

# training settings
task = 'X4_Scale50'
EPOCH = 111840
# start_epo = EPOCH * 0.3
start_epo = EPOCH * 0
end_epo = EPOCH * 0.65
batch_size = 2
l1_weight = 0.75
ssim_weight = 1.1

# data path
data_path = '../../M3FD/Ir_50'
img_list_path = '../../M3FD/Main/train_50.txt'

# save path
pt_path = os.path.join('../train_logs', task, 'pt_root')
subprocess.check_output(['mkdir', '-p', pt_path])

# dataloader
transform = transforms.Compose([transforms.ToTensor()])
convertor = pre_process(data_path, img_list_path, transform)
dataloader = DataLoader(convertor, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

# ng types
transform_types = ['Smoother2', 'LessSaturation2', 'LessContrast3', 'Posterize', 'GaussianBlur', 'DownUpSample3', 'add_new_striped_noise']

# for single gpu
torch.cuda.set_device(1)#All .cuda() that appears after setting it points to a GPU with index of x
ng = ng(transform_types).cuda()
ne = ne().cuda()

# for multi gpu
# ng = ng('train', transform_types).to('cuda:0')
# ng = torch.nn.DataParallel(ng, device_ids=[0, 1])
# ne = ne().to('cuda:0')
# ne = torch.nn.DataParallel(ne, device_ids=[0, 1])

ne.train()

# optim and loss
optimizer_ng = optim.SGD(ng.parameters(), lr=2e-4, momentum=0.9, weight_decay=0.0001)
optimizer_ne = optim.Adam(ne.parameters(), lr=1e-4)
L1_loss = nn.L1Loss()

# training
count = len(convertor)
iter_per_epo = int(count/batch_size)
print("dataset size: {}".format(count))
print('start training...')
for epoch in range(EPOCH):
    for iteration, img in enumerate(dataloader):
        start_time = time.time()
        img = img.cuda()

        # ng
        lq = ng(img)

        # ne
        hq = ne(lq)

        # ssim loss
        ne_ssim_loss = -ssim(hq, img)
        ne_ssim_loss = ne_ssim_loss * ssim_weight

        # L1 loss
        ne_l1_loss = L1_loss(hq, img)
        ne_l1_loss = ne_l1_loss * l1_weight

        # update
        ne_loss = ne_ssim_loss + ne_l1_loss
        optimizer_ne.zero_grad()
        ne_loss.backward()
        optimizer_ne.step()

        if start_epo < epoch < end_epo:
            lq = lq.detach()
            ne_loss = ne_loss.detach().requires_grad_(True)

            # ssim loss
            ng_ssim_loss = ssim(img, lq)
            ng_ssim_loss = ng_ssim_loss * ssim_weight

            # L1 loss
            ng_l1_loss = -L1_loss(img, lq)
            ng_l1_loss = ng_l1_loss * l1_weight

            ng_loss = (ng_ssim_loss + ng_l1_loss) * 0.45 - ne_loss * 0.55
            optimizer_ng.zero_grad()
            ng_loss.backward()
            optimizer_ng.step()

        end_time = time.time()
        cost_time = end_time - start_time

        print('epoch: ' + str(epoch + 1) + '/' + str(EPOCH), end=',  ')
        print('iteration: ' + str(iteration + 1) + '/' + str(iter_per_epo), end=',  ')
        if start_epo < epoch < end_epo:
            print('ng_loss: ' + str('{:.3f}'.format(ng_loss.item())), end=',  ')
        print('ne_loss: ' + str('{:.3f}'.format(ne_loss.item())), end=',  ')
        print('cost_time: ' + str('{:.2f}'.format(cost_time)), end='s,  ')
        print('total_eta: ' + str(int(cost_time * iter_per_epo * (EPOCH - epoch) / 60)), end='min  \n')

    if epoch%10 == 0:
        torch.save(ne.state_dict(), os.path.join(pt_path, 'epoch_' + str(epoch + 1) + '_ne_model.pt'))
        if start_epo < epoch < end_epo:
            torch.save(ng.state_dict(), os.path.join(pt_path, 'epoch_' + str(epoch + 1) + '_ng_model.pt'))