from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os 
import numpy as np
from model import EAST
from dataset import ImageDataSet
from loss import LossFunction
from tqdm import tqdm
import time

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}
train_data_dir = config["train_data_dir"]
mini_batch_size = config["mini_batch_size"]
epochs = config["epochs"]
save_step = config["save_step"]

train_images_dir = os.path.join(train_data_dir, "images")
train_annotations_dir = os.path.join(train_data_dir, "annotations")

trainset = ImageDataSet(train_images_dir, train_annotations_dir)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True)

print("Number of examples:", len(trainset))
print("Mini batch size:", mini_batch_size)
n_mini_batches = len(trainset)//mini_batch_size + int(len(trainset)%mini_batch_size!=0)
print("Number of mini batches:", n_mini_batches) 

model = EAST(geometry=config["geometry"])
loss_function = LossFunction()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

losses = []
score_losses = []
geometry_losses = []
for e in tqdm(range(epochs)):
	model = model.train()
	epoch_loss = 0
	epoch_score_loss = 0
	epoch_geometry_loss = 0

	for i, train_egs in tqdm(enumerate(train_loader), total=n_mini_batches):
		optimizer.zero_grad()

		images, score_maps, geometry_maps = train_egs  
		images = Variable(images.cuda())
		score_maps = Variable(score_maps.cuda())
		geometry_maps = Variable(geometry_maps.cuda())
		print("images", images.size())
		print("score_maps", score_maps.size(), "geometry_maps", geometry_maps.size())

		score_maps_pred, geometry_maps_pred = east.forward(images)
		print("score_maps_pred", score_maps_pred.size(), "geometry_maps_pred", geometry_maps_pred.size())
		
		mini_batch_loss = loss_function.compute_loss(score_maps, score_maps_pred, geometry_maps, geometry_maps_pred)
		print("Score Loss:", loss_function.loss_of_score)
		print("Geometry Loss:", loss_function.loss_of_geometry)
		print("Loss:", mini_batch_loss)
		epoch_loss += mini_batch_loss.item()
		epoch_score_loss += loss_function.loss_of_score.item()
		epoch_geometry_loss += loss_function.loss_of_geometry.item()

		mini_batch_loss.backward()
		optimizer.step()
		scheduler.step()

		time.sleep(5)

	epoch_loss /= n_mini_batches
	epoch_score_loss /= n_mini_batches
	epoch_geometry_loss /= n_mini_batches
	losses.append(epoch_loss)
	score_losses.append(epoch_score_loss)
	geometry_losses.append(epoch_geometry_loss)

	if (e + 1) % save_step == 0:
		if not os.path.exists('./checkpoints'):
			os.mkdir('./checkpoints')
		torch.save(model.state_dict(), './checkpoints/model_{}.pth'.format(e + 1))
    
if not os.path.exists('./plots'):
	os.mkdir('./plots')

plt.plot(losses)
plt.xticks(range(1, epochs+1))
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
plt.savefig('plots/loss.png')

plt.plot(score_losses)
plt.xticks(range(1, epochs+1))
plt.xlabel("epochs")
plt.ylabel("score loss")
plt.show()
plt.savefig('plots/score_loss.png')

plt.plot(geometry_losses)
plt.xticks(range(1, epochs+1))
plt.xlabel("epochs")
plt.ylabel("geometry loss")
plt.show()
plt.savefig('plots/geometry_loss.png')