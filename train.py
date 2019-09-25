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
import matplotlib.pyplot as plt
import csv
import json
import math


config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

geometry = config["geometry"]

train_data_dir = config["train_data_dir"]

cuda = config["cuda"]
lambda_geometry = config["lambda_geometry"]
epochs = config["epochs"]
smoothed_l1_loss_beta = config["smoothed_l1_loss_beta"]
learning_rate = config["learning_rate"]
lr_scheduler_step_size = config['lr_scheduler_step_size']
lr_scheduler_gamma = config['lr_scheduler_gamma']
mini_batch_size = config["mini_batch_size"]
save_step = config["save_step"]

meta_data_dir = config["meta_data_dir"]
model_dir = config["model_dir"]
loss_dir = config["loss_dir"]
plot_dir = config["plot_dir"]
meta_data_file = config["meta_data_file"]
model_file = config["model_file"]
loss_file = config["loss_file"]
plot_file = config["plot_file"]
meta_data = config["meta_data"]

for dir_ in [meta_data_dir, model_dir, loss_dir, plot_dir]:   
    if not os.path.exists(dir_):
        os.mkdir(dir_)
                 
with open(meta_data_file, 'w') as file:
    json.dump(meta_data, file)
with open(loss_file, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch_number', 'mini_batch_number', 'score_loss', 'geometry_loss', 'loss'])
    
train_images_dir = os.path.join(train_data_dir, "images")
train_annotations_dir = os.path.join(train_data_dir, "annotations")

trainset = ImageDataSet(train_images_dir, train_annotations_dir)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True)

n_mini_batches = math.ceil(len(trainset)/mini_batch_size)
print("Number of examples:", len(trainset))
print("Mini batch size:", mini_batch_size)
print("Number of epochs:", epochs)
print("Number of mini batches:", n_mini_batches) 

model = EAST(geometry=geometry)
model = model.train()
loss_function = LossFunction()
if cuda:
    model.cuda()
    loss_function.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

losses = []
score_losses = []
geometry_losses = []
with torch.autograd.set_detect_anomaly(True):
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_score_loss = 0
        epoch_geometry_loss = 0

        tic = time.time()
        for i, train_egs in tqdm(enumerate(train_loader, start=1), total=n_mini_batches, desc="Training Mini Batches:"):
            optimizer.zero_grad()

            images, score_maps, geometry_maps = train_egs  
            if cuda:
                images = Variable(images.cuda())
                score_maps = Variable(score_maps.cuda())
                geometry_maps = Variable(geometry_maps.cuda())
            #print("images", images.size())
            #print("score_maps", score_maps.size(), "geometry_maps", geometry_maps.size())

            score_maps_pred, geometry_maps_pred = model.forward(images)
            #print("score_maps_pred", score_maps_pred.size(), "geometry_maps_pred", geometry_maps_pred.size())

            mini_batch_loss = loss_function.compute_loss(score_maps.double(), 
                score_maps_pred.double(),
                geometry_maps.double(), 
                geometry_maps_pred.double(),
                smoothed_l1_loss_beta = smoothed_l1_loss_beta)
            
            mini_batch_loss_of_score_item = loss_function.loss_of_score.item()
            mini_batch_loss_of_geometry_item = loss_function.loss_of_geometry.item()
            mini_batch_loss_item = mini_batch_loss.item()
            with open(loss_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([e, i, mini_batch_loss_of_score_item, mini_batch_loss_of_geometry_item, mini_batch_loss_item])
            #print("Score Loss:", mini_batch_loss_of_score_item)
            #print("Geometry Loss:", mini_batch_loss_of_geometry_item)
            #print("Loss:", mini_batch_loss_item)

            epoch_score_loss += mini_batch_loss_of_score_item
            epoch_geometry_loss += mini_batch_loss_of_geometry_item
            epoch_loss += mini_batch_loss_item

            mini_batch_loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss /= n_mini_batches
        epoch_score_loss /= n_mini_batches
        epoch_geometry_loss /= n_mini_batches
        losses.append(epoch_loss)
        score_losses.append(epoch_score_loss)
        geometry_losses.append(epoch_geometry_loss)
        toc = time.time()
        elapsed_time = toc - tic
        print("Epoch:{}/{}  Loss:{:.6f}  ScoreLoss:{:.6f}  GeometryLoss:{:.6f}  Duration:{}".format(
            e, 
            epochs,
            epoch_loss, 
            epoch_score_loss, 
            epoch_geometry_loss,
            time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        if (e) % save_step == 0:
            torch.save(model.state_dict(), model_file.format(str(e)))


    plt.figure()
    plt.plot(range(1, epochs+1), losses, marker="o", linestyle="--")
    plt.xticks(range(1, epochs+1))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(plot_file.format('loss'))

    plt.figure()
    plt.plot(range(1, epochs+1), score_losses, marker="o", linestyle="--")
    plt.xticks(range(1, epochs+1))
    plt.xlabel("epochs")
    plt.ylabel("score loss")
    plt.savefig(plot_file.format("score_loss"))

    plt.figure()
    plt.plot(range(1, epochs+1), geometry_losses, marker="o", linestyle="--")
    plt.xticks(range(1, epochs+1))
    plt.xlabel("epochs")
    plt.ylabel("geometry loss")
    plt.savefig(plot_file.format("geometry_loss"))