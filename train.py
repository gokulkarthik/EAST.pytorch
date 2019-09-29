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
from slack import WebClient
from utils import send_message, send_picture

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

geometry = config["geometry"]
label_method = config['label_method']
use_formatted_data = config['use_formatted_data']

use_slack = config["use_slack"]
slack_epoch_step = config["slack_epoch_step"]
slack_channel = config["slack_channel"]

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

experiment_name = config["experiment_name"]
meta_data_dir = config["meta_data_dir"]
model_dir = config["model_dir"]
loss_dir = config["loss_dir"]
plot_dir = config["plot_dir"]
meta_data_file = config["meta_data_file"]
model_file = config["model_file"]
loss_file = config["loss_file"]
plot_file = config["plot_file"]
meta_data = config["meta_data"]

representation = geometry + "_" + label_method

if use_slack and os.environ['SLACK_TOKEN']:
    slack_client = WebClient(os.environ.get('SLACK_TOKEN'))

for dir_ in [meta_data_dir, model_dir, loss_dir, plot_dir]:   
    if not os.path.exists(dir_):
        os.mkdir(dir_)
                 
with open(meta_data_file, 'w') as file:
    json.dump(meta_data, file)
with open(loss_file, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch_number', 'mini_batch_number', 'score_loss', 'geometry_loss', 'loss'])
    
if use_slack:
    message = "Experiment {} started!".format(experiment_name)
    send_message(slack_client, slack_channel, message)
    message = str(meta_data)
    send_message(slack_client, slack_channel, message)
    
train_images_dir = os.path.join(train_data_dir, "images")
train_annotations_dir = os.path.join(train_data_dir, "annotations")
if use_formatted_data:
    train_annotations_dir = train_annotations_dir + "_" + representation

trainset = ImageDataSet(train_images_dir, train_annotations_dir)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True)

n_mini_batches = math.ceil(len(trainset)/mini_batch_size)
print("Number of examples:", len(trainset))
print("Mini batch size:", mini_batch_size)
print("Number of epochs:", epochs)
print("Number of mini batches:", n_mini_batches) 


model = EAST(geometry=geometry, label_method=label_method)
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

            image_names, images, score_maps, geometry_maps = train_egs  
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
        message = "Epoch:{}/{}  ScoreLoss:{:.6f}  GeometryLoss:{:.6f}  Loss:{:.6f}  Duration:{}".format(
            e, 
            epochs,
            epoch_score_loss, 
            epoch_geometry_loss,
            epoch_loss, 
            time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print(message)

        if e % save_step == 0:
            torch.save(model.state_dict(), model_file.format(str(e)))
            keep_n = 1
            file_to_delete = model_file.format(str(e-(keep_n*save_step)))
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                
        if use_slack and e % slack_epoch_step == 0:
            send_message(slack_client, slack_channel, message)
    
    loss_type = "score_loss"
    plt.figure()
    plt.plot(range(1, epochs+1), score_losses, marker="o", linestyle="--")
    plt.xticks(range(1, epochs+1))
    plt.xlabel("epochs")
    plt.ylabel(loss_type)
    plt.savefig(plot_file.format(loss_type))
    if use_slack:
        send_picture(slack_client, slack_channel, loss_type, plot_file.format(loss_type))

    loss_type = "geometry_loss"
    plt.figure()
    plt.plot(range(1, epochs+1), geometry_losses, marker="o", linestyle="--")
    plt.xticks(range(1, epochs+1))
    plt.xlabel("epochs")
    plt.ylabel(loss_type)
    plt.savefig(plot_file.format(loss_type))
    if use_slack:
        send_picture(slack_client, slack_channel, loss_type, plot_file.format(loss_type))
        
    loss_type = "loss"
    plt.figure()
    plt.plot(range(1, epochs+1), losses, marker="o", linestyle="--")
    plt.xticks(range(1, epochs+1))
    plt.xlabel("epochs")
    plt.ylabel(loss_type)
    plt.savefig(plot_file.format(loss_type))
    if use_slack:
        send_picture(slack_client, slack_channel, loss_type, plot_file.format(loss_type))
        send_message(slack_client, slack_channel, message=":tada: "*5)
        