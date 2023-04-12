import os
import argparse
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='path to dataset', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--TFD_checkpoint_dir', default='path to checkpoint of TFD', type=str, help='checkpoint of TFD')
parser.add_argument('--LER_checkpoint_dir', default='path to checkpoint of LER', type=str, help='checkpoint of LER')
parser.add_argument('--dataset', default='NTIRE2023', type=str, help='dataset name')
parser.add_argument('--gpu', default='0, 1', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network_TFD, network_LER, result_dir, lpips_metric):
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	LPIPS = AverageMeter()

	torch.cuda.empty_cache()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		patches = batch['source'].cuda()
		target = batch['target'].cuda()
		filename = batch['filename'][0]


		with torch.no_grad():
			_, dehaze_reconstruct, x_J, _ = network_TFD(patches)
			out, _ = network_LER(dehaze_reconstruct, x_J)
			out = out.clamp_(-1, 1)
			output = out * 0.5 + 0.5  # [-1, 1] to [0, 1]

		with torch.no_grad():
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
							data_range=1, size_average=False).item()

			lpips_val = lpips_metric(output, target)

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)
		LPIPS.update(lpips_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.04f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.04f})\t'
			  'LPIPS: {lpips.val:.03f} ({lpips.avg:.04f})\t'
			  'Filename: {s}'.format(idx, psnr=PSNR, ssim=SSIM, lpips=LPIPS, s=filename))

		f_result.write('%s,%.03f,%.03f,%.03f\n'%(filename, psnr_val, ssim_val, lpips_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'),
			  os.path.join(result_dir, '%.04f | %.04f | %.04f.csv'%(PSNR.avg, SSIM.avg, LPIPS.avg)))


if __name__ == '__main__':
	network_TFD = TFD()
	network_TFD = network_TFD.cuda()
	network_LER = LER()
	network_LER = network_LER.cuda()


	network_TFD.load_state_dict(single(args.TFD_checkpoint_dir))
	network_LER.load_state_dict(single(args.LER_checkpoint_dir))

	lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	test_dataset = PairLoader(dataset_dir, 'test', mode='test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network_TFD, network_LER, result_dir, lpips_metric)