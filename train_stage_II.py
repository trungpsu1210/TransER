import argparse
import json
import torch
from torchvision.models import vgg16
from pytorch_msssim import ssim, MS_SSIM
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import time
from utils import *
from perceptual import LossNetwork
from data_loader import *
from models import *

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='path to dataset', type=str, help='path to dataset')
parser.add_argument('--TFD_checkpoint_dir', default='path to checkpoint of TFD', type=str, help='checkpoint of TFD')
parser.add_argument('--TRN_checkpoint_dir', default='path to checkpoint of TRN', type=str, help='checkpoint of TRN')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='NTIRE2023', type=str, help='dataset name')
parser.add_argument('--gpu', default='0, 1', type=str, help='GPUs used for training')
parser.add_argument('--resume', type=bool, default=True, help='Continue Train')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network_TFD, network_TRN, network_LER, criterion, optimizer, scaler):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network_LER.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):

            _, dehaze_reconstruct, x_J, _ = network_TFD(source_img)
            output, x_mid_student = network_LER(dehaze_reconstruct, x_J)
            _, x_mid_teacher = network_TRN(target_img, target_img)

            loss_KT = criterion[0](x_mid_teacher, x_mid_student)

            loss_rec = criterion[0](output, target_img)
            loss_pec = criterion[2](output, target_img)
            loss_ssim = - criterion[1](output.float(), target_img)

            loss = loss_rec + loss_KT + 0.2 * loss_ssim + 0.1 * loss_pec

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network_TFD, network_LER, lpips_metric):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS = AverageMeter()

    torch.cuda.empty_cache()

    network_TFD.eval()
    network_LER.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()


        with torch.no_grad():

            _, dehaze_reconstruct, x_J, _ = network_TFD(source_img)
            output, _ = network_LER(dehaze_reconstruct, x_J)
            output = output.clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))
        ssim_value = ssim(F.adaptive_avg_pool2d(output * 0.5 + 0.5, (int(H / down_ratio), int(W / down_ratio))),
                          F.adaptive_avg_pool2d(target_img * 0.5 + 0.5, (int(H / down_ratio), int(W / down_ratio))),
                          data_range=1, size_average=False).mean()
        SSIM.update(ssim_value.item(), source_img.size(0))

        with torch.no_grad():
            lpips = lpips_metric(output, target_img)
            LPIPS.update(lpips, source_img.size(0))

    return PSNR.avg, SSIM.avg, LPIPS.avg


if __name__ == '__main__':

    setting_filename = 'configs.json'
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network_TFD = TFD()
    network_TFD = nn.DataParallel(network_TFD).cuda()

    network_TRN = TRN()
    network_TRN = nn.DataParallel(network_TRN).cuda()

    network_LER = LER()
    network_LER = nn.DataParallel(network_LER).cuda()

    start_time = time.time()

    # Loss
    vgg_model = vgg16(pretrained=True)
    vgg_model = vgg_model.features[:16].cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    criterion = []
    criterion.append(nn.L1Loss().cuda())
    criterion.append(MS_SSIM().cuda())
    criterion.append(loss_network)

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network_LER.parameters(), lr=setting['lr_LER'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network_LER.parameters(), lr=setting['lr_LER'])
    else:
        raise Exception("ERROR: unsupported optimizer")


    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_mult=2, last_epoch=-1,
                                                                     T_0=setting['step_T'],
                                                                     eta_min=setting['lr_LER'] * 1e-3)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size_Stage_II'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size_Stage_II'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

    ckp_TFD = torch.load(args.TFD_checkpoint_dir)
    ckp_TRN = torch.load(args.TRN_checkpoint_dir)
    ckp_LER = torch.load(args.TRN_checkpoint_dir)

    try:
        network_TFD.load_state_dict(ckp_TFD['state_dict'])
        network_TRN.load_state_dict(ckp_TRN['state_dict'])
        network_LER.load_state_dict(ckp_LER['state_dict'])

        print('Successfully loading weight to models')
    except:
        print('No weight loaded')

    for name, param in network_TFD.named_parameters():
        param.requires_grad = False

    for name, param in network_TRN.named_parameters():
        param.requires_grad = False

    losses = []
    start_epoch = 0
    best_psnr = 0
    best_ssim = 0
    best_lpips = 1
    best = 0
    psnrs = []
    ssims = []
    lpips = []

    if args.resume and os.path.exists(os.path.join(save_dir, 'LER.pth')):
        ckp = torch.load(os.path.join(save_dir, 'LER.pth'))
        network_LER.load_state_dict(ckp['state_dict'])
        start_epoch = ckp['epoch']
        best_psnr = ckp['best_psnr']
        best_ssim = ckp['best_ssim']
        best_lpips = ckp['best_lpips']
        ssims = ckp['ssims']
        psnrs = ckp['psnrs']
        lpips = ckp['lpips']
        print(f'start_step: {start_epoch} continue to train ---')
    else:
        print('==> Start training from scratch, current model name: ' + args.model)

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

    for epoch in range(start_epoch + 1, setting['epochs_LER'] + 1):
        loss = train(train_loader, network_TFD, network_TRN, network_LER, criterion, optimizer, scaler)
        losses.append(loss)
        writer.add_scalar('train_loss', loss, epoch)
        print(
            f'\rTrain loss: {loss:.5f} | epoch: {epoch} | lr: {optimizer.param_groups[0]["lr"]:.6f} | time_used: {(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)
        scheduler.step()

        if epoch % setting['eval_freq'] == 0:
            avg_psnr, avg_ssim, avg_lpips = valid(val_loader, network_TFD, network_LER, lpips_metric)

            print(f'\nEpoch: {epoch} | psnr: {avg_psnr:.4f} | ssim: {avg_ssim:.4f} | lpips: {avg_lpips:.4f}')
            writer.add_scalar('valid_psnr', avg_psnr, epoch)
            writer.add_scalar('valid_ssim', avg_ssim, epoch)
            writer.add_scalar('valid_lpips', avg_lpips, epoch)

            psnrs.append(avg_psnr)
            ssims.append(avg_ssim)
            lpips.append(avg_lpips)

            # Depend on which metrics to set up the saved checkpoint. Default is set to overall performance
            if (avg_ssim > best_ssim) and (avg_psnr > best_psnr) and (avg_lpips < best_lpips):

                best_psnr = avg_psnr
                best_ssim = avg_ssim
                best_lpips = avg_lpips

                torch.save({'epoch': epoch,
                            'best_psnr': best_psnr,
                            'best_ssim': best_ssim,
                            'best_lpips': best_lpips,
                            'psnrs': psnrs,
                            'ssims': ssims,
                            'lpips': lpips,
                            'state_dict': network_LER.state_dict()},
                           os.path.join(save_dir, 'LER.pth'))

                print(
                    f'\n Models saved at epoch: {epoch} | best_psnr: {best_psnr:.4f} | best_ssim: {best_ssim:.4f} | | best_lpips: {best_lpips:.4f}')

    print(
        f'\nFinished Training Model | best_psnr: {best_psnr:.4f} | best_ssim: {best_ssim:.4f} | best_lpips: {best_lpips:.4f}')
