import os

class Config:

	user = "gokul"
	lang = "hindi"
	geometry = "QUAD" # ["RBOX", "QUAD"]

	max_m_train = 10000
	data_dir = "/home/{}/data-split/{}".format(user, lang)
	train_data_dir = os.path.join(data_dir, 'train')
    dev_data_dir = os.path.join(data_dir, 'dev')
    test_data_dir = os.path.join(data_dir, 'test')

    epochs = 5
    learning_rate = 0.01
    mini_batch_size = 2