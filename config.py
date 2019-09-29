import os

class Config:

    user = "gokul"
    lang = "hindi"

    image_size = [512, 512, 3]
    geometry = "QUAD" # ["RBOX", "QUAD"]
    label_method = "multiple" # ["single", "multiple"]
    use_formatted_data = True
    
    use_slack = True
    slack_epoch_step = 1
    slack_channel = "#updates" # "CNU04UXUN" 

    max_m_train = 120
    data_dir = "/home/{}/data-split/{}".format(user, lang)
    train_data_dir = os.path.join(data_dir, 'train')
    dev_data_dir = os.path.join(data_dir, 'dev')
    test_data_dir = os.path.join(data_dir, 'test')

    cuda = True
    lambda_score = 1
    lambda_geometry = 1
    epochs = 50
    smoothed_l1_loss_beta = 1.0
    learning_rate = 0.005
    lr_scheduler_step_size = 2000 # for every 4 epochs
    lr_scheduler_gamma = .94
    mini_batch_size = 24
    save_step = 5
    
    experiment_name = "dummy"
    meta_data_dir = "./experiment_meta_data" # 1
    model_dir = "./experiment_model" # epochs/save_step
    loss_dir = "./experiment_loss" # 1
    plot_dir = "./experiment_plot" # 3
    meta_data_file = os.path.join(meta_data_dir, "experiment_{}.json".format(experiment_name))
    model_file = os.path.join(model_dir, "experiment_" + experiment_name + "_epoch_{}.pth") # format during train
    loss_file = os.path.join(loss_dir, "experiment_{}.csv".format(experiment_name))
    plot_file = os.path.join(plot_dir, "experiment_" + experiment_name + "_{}.png") # format during train             

    meta_data = {"geometry":geometry,
                 "max_m_train":max_m_train,
                 "lambda_score":lambda_score,
                 "lambda_geometry":lambda_geometry,
                 "epochs":epochs, 
                 "smoothed_l1_loss_beta": smoothed_l1_loss_beta,
                 "learning_rate":learning_rate,
                 "lr_scheduler_step_size": lr_scheduler_step_size,
                 "lr_scheduler_gamma": lr_scheduler_gamma,
                 "mini_batch_size":mini_batch_size,
                 "comments": "Model: xavier init; Score Loss: cross entropy ;   Geo Loss: L1 loss with text mask normalized by 8*512"
                }
    
    trained_model_file = "./experiment_model/experiment_{}_epoch_{}.pth".format("4", "25") 
    eval_mini_batch_size = 16
    test_mini_batch_size = 16
    
    score_threshold = 0
    iou_threshold = 0.1
    max_boxes = 5