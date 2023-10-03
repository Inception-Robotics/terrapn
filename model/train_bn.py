import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model_bn import get_model
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from os.path import expanduser
import argparse


fd_name = "weights_bn6"

def get_data(data_dir, v_len=50):
    img_data = [] 
    vel_data = []
    labels = []
    image_count = 0
    print(data_dir)
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if name.endswith(".png"):
                image_file = data_dir + "/" + name
                i = Image.open(image_file)
                i = np.asarray(i)
                img_data.append(i)
                print(name.split(".")[0])
                vel_file = data_dir + "input_velocity_" + str(name.split(".")[0]) + ".npy"
                label_file = data_dir + "label_" + str(name.split(".")[0]) + ".npy"
                print(label_file)
                print(vel_file)
                v = np.load(vel_file)[-1 * v_len:]
                vel_data.append(v.reshape(-1))
                ll = np.load(label_file)
                labels.append(ll)
                image_count = image_count + 1
    return np.array(img_data), np.array(vel_data), np.array(labels), image_count


def main():
    parser = argparse.ArgumentParser(description='Training script to train TerraPN')
    parser.add_argument('--v_len', type=int, help='Length of velocity commands', default=50)
    parser.add_argument('--patch_len', type=int, help='Size of the image patch', default=100)
    parser.add_argument('--training_type', type=str, help='[train]Do you want to train from scratch or [retrain]retrain the existing model?', default="retrain")
    parser.add_argument('--data_split', type=float, help='Train and test data split in fraction', default=0.1)
    args=parser.parse_args()
    v_len = args.v_len
    patch_len = args.patch_len
    seed = 0
    dataset_dir = expanduser("~") + "/inception/labeled_dataset/"
    if not os.path.exists(expanduser("~") + "/inception/" + "trained_model"):
        os.makedirs(expanduser("~") + "/inception/" + "trained_model")
    trained_model_dir = expanduser("~") + "/inception/" + "trained_model/"
    img_data, vel_data, labels, image_count = get_data(dataset_dir, v_len)
    training_type = args.training_type
    model_name = "Weights-047--0.40830.hdf5"
    test_data_split = args.data_split # In fraction

    np.random.seed(seed)  
    rand = np.arange(image_count)
    np.random.shuffle(rand)
    split_point = int(image_count * test_data_split)

    img_train = img_data[rand[split_point:]]
    vel_train = vel_data[rand[split_point:]]
    labels_train = labels[rand[split_point:]]
    img_test = img_data[rand[:split_point]]
    vel_test = vel_data[rand[:split_point]]
    labels_test = labels[rand[:split_point]]


    if training_type == "train":
        print("---------------------Training from scratch-----------------------")
        model = get_model(v_len, patch_len)
        
    elif training_type == "retrain":
        model = load_model(model_name)
        print("+++++++++++++++++++++ Retraining the existing model +++++++++++++++++++++++")
    checkpoint_name = trained_model_dir + '/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    csv_logger = CSVLogger(trained_model_dir+ '/training.log')
    callbacks_list = [checkpoint, csv_logger]
    model.fit([img_train, vel_train], labels_train, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)



if __name__ == '__main__':
	main()
