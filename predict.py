import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from utils import chw_to_hwc, SingleLoaderEMPatches
from models import *
from empatches import EMPatches

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', default='./results/', type=str, help='path to saved results')
parser.add_argument('--data_dir', default='path to dataset', type=str, help='path to dataset')
parser.add_argument('--TFD_checkpoint_dir', default='path to checkpoint of TFD', type=str, help='checkpoint of TFD')
parser.add_argument('--LER_checkpoint_dir', default='path to checkpoint of LER', type=str, help='checkpoint of LER')
args = parser.parse_args()


def test(test_loader, network_TFD, netowrk_LER, result_dir):
	torch.cuda.empty_cache()

	os.makedirs(result_dir, exist_ok=True)

	for batch in tqdm(test_loader):
		patches = batch['patches']
		indices = batch['indices']
		filename = batch['filename'][0]

		for i in range(len(indices)):

			patches[i] = patches[i].cuda()
			with torch.no_grad():
				_, dehaze_re, x_J, _ = network_TFD(patches[i])
				patches[i], _ = netowrk_LER(dehaze_re, x_J)
				patches[i] = patches[i].clamp_(-1, 1)
				patches[i] = patches[i] * 0.5 + 0.5

			patches[i] = chw_to_hwc(patches[i].detach().cpu().squeeze(0).numpy())

		emp = EMPatches()

		out_img = emp.merge_patches(patches, indices, mode='avg')

		out_img = (out_img * 255.0).astype('uint8')
		out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGBA)
		cv2.imwrite(os.path.join(result_dir, filename), out_img)


if __name__ == '__main__':

	network_TFD = TFD()
	network_TFD = network_TFD.cuda()

	netowrk_LER = LER()
	netowrk_LER = netowrk_LER.cuda()

	network_TFD.load_state_dict(torch.load(args.TFD_checkpoint_dir)['state_dict'])
	netowrk_LER.load_state_dict(torch.load(args.LER_checkpoint_dir)['state_dict'])

	test_dataset = SingleLoaderEMPatches(args.data_dir, patchsize=1000, stride=450)

	test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

	test(test_loader, network_TFD, netowrk_LER, args.result_dir)

