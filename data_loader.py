import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class ImageFolder(data.Dataset):
	def __init__(self, root, image_size=224, mode='train', augmentation_prob=0):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		if root[-6:] == "train/":
			self.GT_paths = root[:-6]+'annotation/'
		elif root[-6:] == "valid/":
			self.GT_paths = root[:-6] + 'annotation/'
		elif root[-5:] == "test/":
			self.GT_paths = root[:-5] + 'annotation/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		if self.mode == 'train' or self.mode == 'valid':
			filename = image_path.split('/')[-1][:-len(".png")]

			GT_path = self.GT_paths + filename + '.png'

			image = Image.open(image_path)
			GT = Image.open(GT_path)

			Transform = []

			Transform.append(T.ToTensor())
			Transform = T.Compose(Transform)

			image = Transform(image)
			GT = Transform(GT)

			Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			image = Norm_(image)
			return image, GT
		elif self.mode == 'test':
			image = Image.open(image_path)
			Transform = []
			Transform.append(T.ToTensor())
			Transform = T.Compose(Transform)

			image = Transform(image)

			Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			image = Norm_(image)
			return [image_path, image]
	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
	if mode == 'train' or mode == 'valid':
		data_loader = data.DataLoader(dataset=dataset,
									  batch_size=batch_size,
									  shuffle=True,
									  num_workers=num_workers)
		# for i, (images, GT) in enumerate(data_loader):
		# 	print(images.shape, GT.shape)
		return data_loader
	elif mode == 'test':
		data_loader = data.DataLoader(dataset=dataset,
									  batch_size=batch_size,
									  num_workers=num_workers)
		# for i, (images_path, images) in enumerate(data_loader):
		# 	print(images_path, images.shape)
		return data_loader