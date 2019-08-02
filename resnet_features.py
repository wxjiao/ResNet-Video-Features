import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from overrides import overrides
import os
import cv2
import pickle
from tqdm import tqdm


# Setup proxy
os.environ['http_proxy'] = 'http://proxy.cse.cuhk.edu.hk:8000'
os.environ['HTTP_PROXY'] = 'http://proxy.cse.cuhk.edu.hk:8000'
os.environ['https_proxy'] = 'http://proxy.cse.cuhk.edu.hk:8000'
os.environ['HTTPS_PROXY'] = 'http://proxy.cse.cuhk.edu.hk:8000'

# Device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the pretrained model
resnet152 = models.resnet152(pretrained=True)
resnet152.eval()
resnet152.to(DEVICE)

# Block fc layer
class Identity(nn.Module):
	@overrides
	def forward(self, input_):
		return input_

resnet152.fc = Identity()

# Image transforms
transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_frames(frames_folder_path):
	# Get all frame file names
	frames = None
	frames_file = os.listdir(frames_folder_path)
	for i,frame_file_name in enumerate(frames_file):
		frame = Image.open(os.path.join(frames_folder_path, frame_file_name))
		frame = transf(frame)

		if frames is None:
			frames = torch.empty((len(frames_file), *frame.size()))

		frames[i] = frame

	return frames


def frames_features(frames_folder_path):
	frames = get_frames(frames_folder_path)
	frames = frames.to(DEVICE)
	# Run the model on input data
	output = []
	batch_size = 10                 # 10 for PC
	for start_index in range(0, len(frames), batch_size):
		end_index = min(start_index + batch_size, len(frames))
		frame_range = range(start_index, end_index)
		frame_batch = frames[frame_range]
		avg_pool_value = resnet152(frame_batch)
		output.append(avg_pool_value.detach().cpu().numpy())

	output = np.concatenate(output)

	return output


def videos_features(frames_folders_path, videos_path, save_path):
	# frames_folders_path: path to all video frames folders
	# video_path: path ot original videos
	frames_folders = os.listdir(frames_folders_path)
	features = {}
	for frames_folder in tqdm(frames_folders, ncols=100, ascii=True):
		video_feature = {}
		video_name = os.path.join(videos_path, frames_folder + '.mp4')
		frames_folder_path = os.path.join(frames_folders_path, frames_folder)
		cam = cv2.VideoCapture(video_name)
		fps = round(cam.get(cv2.CAP_PROP_FPS), 0)
		feat = frames_features(frames_folder_path)
		video_feature['fps'] = fps
		video_feature['resnet152'] = feat
		features[frames_folder] = video_feature
		#print("Process video {} FPS {} shape {}".format(frames_folder, fps, feat.shape))

	with open(save_path, 'wb') as f:
		pickle.dump(features, f)


def main():
	frames_folders_path = "./Frames_folders"
	videos_path = "./Videos"
	videos_features(frames_folders_path, videos_path, 'resnet_video_features.pt')


if __name__ == '__main__':
	main()
