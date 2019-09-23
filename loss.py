from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}
geometry = config['geometry']
label_method = config['label_method']
image_size = config['image_size']
n_H, n_W, n_C = image_size
lambda_geometry = config['lambda_geometry']

if geometry == "RBOX":
    raise NotImplementedError("Only implemented for the QUAD geometry")
if label_method == "multiple":
    raise NotImplementedError("Only implemented for the single label method")

class LossFunction(nn.Module):

	def __init__(self):

		super(LossFunction, self).__init__()
		self.loss_of_score = None
		self.loss_of_geometry = None
		self.loss = None


	def compute_geometry_beta(self, y_true_geometry_cell):
		
		D = []
		for i in range(0, 8, 2): # 0,2,4,6
			indices = [i, i+1, (i+2)%8, (i+3)%8]
			x1, y1, x2, y2 = y_true_geometry_cell[indices]
			d = (x1 - x2) ** 2 + (y1 - y2) ** 2
			D.append(d)
		D = torch.Tensor(D)

		return torch.sqrt(torch.min(D))

	def compute_smoothed_l1_loss(self, y_true, y_pred):
		
		"""
		y_true, y_pred: [8]
		"""
		l1_loss = torch.abs(y_true - y_pred)
		smoothed_l1_loss = 0
		for l1 in l1_loss:
			if l1 < 1:
				l1 = 0.5 * l1 * l1
			else:
				l1 -= 0.5
			smoothed_l1_loss += l1

		return smoothed_l1_loss


	def compute_score_loss(self, Y_true_score, Y_pred_score):
		
		"""
		y_true_score, y_pred_score: [m, 1, 128, 128]; range: [0,1]
		"""
		
		m = Y_true_score.shape[0]
		beta = 1 - (Y_true_score.sum()/torch.numel(Y_true_score))
		loss_of_score_pos = -beta * Y_true_score * torch.log(Y_pred_score) # [m, 1, 128, 128]
		loss_of_score_neg = -(1 - beta) * (1 - Y_true_score) * torch.log(1 - Y_pred_score) # [m, 1, 128, 128]
		loss_of_score = torch.sum(loss_of_score_pos + loss_of_score_neg) / m

		return loss_of_score


	def compute_geometry_loss(self, Y_true_geometry, Y_pred_geometry, lamda_geometry):

		"""
		y_true_geometry, y_pred_geometry: [m, 8, 128, 128]; range:[0,1]
		beta: N_Q*
		"""
		m, n_c, n_H, n_W = Y_true_geometry.shape
		loss_of_geometry = 0
		for y_true_geometry, y_pred_geometry in zip(Y_true_geometry, Y_pred_geometry):
			for h in range(n_H):
				for w in range(n_W):
					y_true_geometry_cell = y_true_geometry[:, h, w]
					y_pred_geometry_cell = y_pred_geometry[:, h, w]
					#geometry_beta = self.compute_geometry_beta(y_true_geometry_cell)
					#print("geometry_beta:", geometry_beta.item())
					loss = self.compute_smoothed_l1_loss(y_true_geometry_cell, y_pred_geometry_cell) / 8.0
					#loss /= geometry_beta
					loss_of_geometry += loss

		loss_of_geometry /= float(m * n_H * n_W)
		loss_of_geometry *= lambda_geometry

		return loss_of_geometry



	def compute_loss(self, Y_true_score, Y_pred_score, Y_true_geometry, Y_pred_geometry):
		"""
		y_true_score, y_pred_score: [m, 1, 128, 128]
		y_true_geometry, y_pred_geometry: [m, 8, 128, 128]
		"""
		self.loss_of_score = self.compute_score_loss(Y_true_score, Y_pred_score)
		print("Y_true_geometry.max():", torch.max(Y_true_geometry))
		print("Y_pred_geometry.max():", torch.max(Y_pred_geometry))
		self.loss_of_geometry = self.compute_geometry_loss(Y_true_geometry, Y_pred_geometry, lambda_geometry)
		self.loss = self.loss_of_score + self.loss_of_geometry
		return self.loss

# test code
"""
loss_function = LossFunction()
Y_true_score = torch.rand([2, 1, 128, 128])
Y_pred_score = torch.rand([2, 1, 128, 128])
Y_true_geometry = torch.rand([2, 8, 128, 128])
Y_pred_geometry = torch.rand([2,8, 128, 128])
loss = loss_function.compute_loss(Y_true_score, Y_pred_score, Y_true_geometry, Y_pred_geometry)
print("Score Loss:", loss_function.loss_of_score)
print("Geometry Loss:", loss_function.loss_of_geometry)
print("Loss:", loss)
"""
