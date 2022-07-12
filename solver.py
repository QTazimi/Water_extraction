import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net, AttU_Net
from torch import optim, nn
import cv2
import matplotlib.pyplot as plt
import csv


class Solver(object):
	def __init__(self, config, train_loader=None, valid_loader=None, test_loader=None):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = DiceLoss() #nn.BCEWithLogitsLoss() #torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#:1
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=1)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=3,output_ch=1)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)
		self.unet = nn.DataParallel(self.unet)
		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))


	# def to_data(self, x):
	# 	"""Convert variable to tensor."""
	# 	if torch.cuda.is_available():
	# 		x = x.cpu()
	# 	return x.data

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pth' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			f = open(os.path.join(self.result_path, 'train.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow(["model_type", "Epoch", "Total_Epoch", "loss", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "IOU"])
			f.close()
			f = open(os.path.join(self.result_path, 'valid.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow(["model_type", "Epoch", "Total_Epoch", "acc", "SE", "SP", "PC", "F1", "JS", "DC", "IOU"])
			f.close()
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				IOU = 0.	# Intersection-over-Union, IoU
				# FWIOU = 0.  #Frequency Weighted Intersection-over-Union, FWIoU
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth

					images = images.to(self.device, dtype=torch.float32)
					GT = GT.to(self.device, dtype=torch.float32)
					# SR : Segmentation Result
					SR = self.unet(images)

					SR_flat = SR.view(SR.size(0), -1)#sigmoid

					GT_flat = GT.view(GT.size(0), -1)

					# GT_flat = GT_flat.float()
					loss = self.criterion(SR_flat, GT_flat)
					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					SR_probs = F.sigmoid(SR)
					acc += get_accuracy(SR_probs, GT)
					SE += get_sensitivity(SR_probs, GT)
					SP += get_specificity(SR_probs, GT)
					PC += get_precision(SR_probs, GT)
					F1 += get_F1(SR_probs, GT)
					JS += get_JS(SR_probs, GT)
					DC += get_DC(SR_probs, GT)
					IOU += get_iou(SR_probs, GT)
					# FWIOU += get_FWiou(SR_probs, GT)
					length += 1#images.size(0)

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				IOU = IOU/length
				# FWIOU = FWIOU/length
				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, IOU: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss / length,\
					  acc,SE,SP,PC,F1,JS,DC,IOU))

				f = open(os.path.join(self.result_path, 'train.csv'), 'a', encoding='utf-8', newline='')
				wr = csv.writer(f)
				wr.writerow([self.model_type, epoch+1, self.num_epochs,
							 epoch_loss / length, acc, SE, SP, PC, F1, JS, DC,IOU])
				f.close()

				# decay_rate = 0.7
				# lr = self.lr * np.power(decay_rate, epoch)
				# print(lr)
				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					# decay_rate = 0.7
					# lr = self.lr * np.power(decay_rate, (epoch + 1 - self.num_epochs_decay) / 20)
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print('Decay learning rate to lr: {}.'.format(lr))


				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				IOU = 0.	# Intersection-over-Union, IoU
				# FWIOU = 0.  #Frequency Weighted Intersection-over-Union, FWIoU
				length = 0
				for i, (images, GT) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = F.sigmoid(self.unet(images))
					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
					IOU += get_iou(SR,GT)
					# FWIOU += get_FWiou(SR,GT)


					length += 1 #images.size(0)
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				IOU = IOU / length
				# FWIOU = FWIOU/length
				unet_score = JS + DC

				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, IOU: %.4f'%(acc,SE,SP,PC,F1,JS,DC,IOU))
				f = open(os.path.join(self.result_path, 'valid.csv'), 'a', encoding='utf-8', newline='')
				wr = csv.writer(f)
				wr.writerow([self.model_type, epoch + 1, self.num_epochs,
							  acc, SE, SP, PC, F1, JS, DC, IOU])
				f.close()
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''

				# print('qsave:', unet_score, best_unet_score)
				# Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet, unet_path)


	def test(self):
		# #===================================== Test ====================================#
		unet_path = "/home/program/Unet/models/U_Net-350-0.0002-70-0.0000.pth"
		save_path = ''
		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))
		self.unet.eval()
		# acc = 0.  # Accuracy
		# SE = 0.  # Sensitivity (Recall)
		# SP = 0.  # Specificity
		# PC = 0.  # Precision
		# F1 = 0.  # F1 Score
		# JS = 0.  # Jaccard Similarity
		# DC = 0.  # Dice Coefficient
		# length = 0

		for i, (image_path, image) in enumerate(self.test_loader):
			image = image.to(device=self.device, dtype=torch.float32)
			pred = F.sigmoid(self.unet(image))
			pred = np.array(pred.data.cpu())
			pred[pred >= 0.5] = 255
			pred[pred < 0.5] = 0
			for image_path_item, pred_item in zip(image_path, pred):
				#找轮廓
				image_path_item = image_path_item
				pred_item = np.array(pred_item, np.uint8)
				pred_item =	pred_item.reshape(pred_item.shape[1], pred_item.shape[2])
				contours, _ = cv2.findContours(pred_item, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				img = cv2.imread(image_path_item, 1)
				cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

				img = img[:, :, ::-1]
				img[..., 2] = np.where(pred_item == 255, 200, img[..., 2])

				plt.imshow(img)
				plt.show()


				# print(image_path)
				# filename = image_path.split('/')[-1][:-len(".png")]
				# save_path = save_path + filename + '.png'
				# cv2.imwrite(save_path, pred)

			#指标计算
			# acc += get_accuracy(SR, GT)
			# SE += get_sensitivity(SR, GT)
			# SP += get_specificity(SR, GT)
			# PC += get_precision(SR, GT)
			# F1 += get_F1(SR, GT)
			# JS += get_JS(SR, GT)
			# DC += get_DC(SR, GT)
			#
			# length += 1  # images.size(0)

		# acc = acc / length
		# SE = SE / length
		# SP = SP / length
		# PC = PC / length
		# F1 = F1 / length
		# JS = JS / length
		# DC = DC / length
		# unet_score = JS + DC
		#
		# f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
		# wr = csv.writer(f)
		# wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC])
		# f.close()


			
