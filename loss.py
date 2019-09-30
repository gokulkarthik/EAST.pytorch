from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}
geometry = config['geometry']
label_method = config['label_method']
image_size = config['image_size']
n_H, n_W, n_C = image_size
lambda_score = config['lambda_score']
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


    def compute_score_loss(self, Y_true_score, Y_pred_score):

        """
        y_true_score, y_pred_score: [m, 1, 128, 128]; range: [0,1]
        """

        m = Y_true_score.shape[0]
        n_cells = torch.numel(Y_true_score)
        n_pos_cells = Y_true_score.sum()
        n_neg_cells = n_cells - n_pos_cells
        beta = 1 - (Y_true_score.sum()/torch.numel(Y_true_score)) # ratio of 0s
        loss_of_score_pos = -beta * Y_true_score * torch.log(Y_pred_score) # [m, 1, 128, 128]
        loss_of_score_neg = -(1 - beta) * (1 - Y_true_score) * torch.log(1 - Y_pred_score) # [m, 1, 128, 128]
        normalization_factor = (beta * n_pos_cells) + ((1-beta)* n_neg_cells)
        loss_of_score = torch.sum(loss_of_score_pos + loss_of_score_neg) / normalization_factor

        return loss_of_score


    def compute_geometry_loss(self, Y_true_geometry, Y_pred_geometry, Y_true_score, smoothed_l1_loss_beta=1):

        """
        Y_true_geometry, Y_pred_geometry: [m, 8, 128, 128]; range:[0,1]
        beta: N_Q*
        Y_true_score: [m, 1, 128, 128]
        """
        beta = smoothed_l1_loss_beta
        diff = torch.abs(Y_true_geometry*Y_true_score - Y_pred_geometry*Y_true_score) # multiply with text mask
        diff = diff / 512
        #diff = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        loss_of_geometry = diff.sum()
        loss_of_geometry /= float(Y_true_score.sum()*8)

        return loss_of_geometry



    def compute_loss(self, Y_true_score, Y_pred_score, Y_true_geometry, Y_pred_geometry, smoothed_l1_loss_beta=1):
        """
        y_true_score, y_pred_score: [m, 1, 128, 128]
        y_true_geometry, y_pred_geometry: [m, 8, 128, 128]
        """
        #print("Y_true_geometry.max():", torch.max(Y_true_geometry).item())
        #print("Y_pred_geometry.max():", torch.max(Y_pred_geometry).item())
        self.loss_of_score = self.compute_score_loss(Y_true_score, Y_pred_score)
        self.loss_of_geometry = self.compute_geometry_loss(Y_true_geometry, 
                                                           Y_pred_geometry, 
                                                           Y_true_score, 
                                                           smoothed_l1_loss_beta=smoothed_l1_loss_beta)
        self.loss = lambda_score * self.loss_of_score + lambda_geometry * self.loss_of_geometry
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
