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
                vel_file = data_dir + "/" + "input_velocity_" + str(name.split(".")[0]) + ".npy"
                label_file = data_dir + "/" + "label_" + str(name.split(".")[0]) + ".npy"
                print(label_file)
                print(vel_file)
                v = np.load(vel_file)[-1 * v_len:]
                vel_data.append(v.reshape(-1))
                ll = np.load(label_file)
                labels.append(ll)
                image_count = image_count + 1
    return np.array(img_data), np.array(vel_data), np.array(labels), image_count



v_len = 50
patch_len = 100
seed = 0
# img_data, vel_data, labels = get_data("/home/rayguan/Desktop/self_sup/dataset/patch100_lst.txt", v_len)
img_data, vel_data, labels, image_count = get_data("/home/gamma-nuc/inception/labeled_dataset", v_len)
train = True
model_lst = "/home/gamma-robot/Downloads/self_sup/Code_Results/weights/Weights-190--0.21869.hdf5"
test_data_split = 0.1 # In fraction

np.random.seed(seed)  
rand = np.arange(image_count)
np.random.shuffle(rand)

img_train = img_data[rand[int(image_count * test_data_split):]]
vel_train = vel_data[rand[int(image_count * test_data_split):]]
labels_train = labels[rand[int(image_count * test_data_split):]]
img_test = img_data[rand[:int(image_count * test_data_split)]]
vel_test = vel_data[rand[:int(image_count * test_data_split)]]
labels_test = labels[rand[:int(image_count * test_data_split)]]

# model = get_model(v_len, patch_len)
#checkpoint_name = './weights/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]


# model.fit([img_train, vel_train], labels_train, epochs=200, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

if train:
    model = get_model(v_len, patch_len)
    checkpoint_name = '/home/gamma-nuc/inception/labeled_dataset' + '/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    csv_logger = CSVLogger('/home/gamma-nuc/inception/labeled_dataset' + '/training.log')
    callbacks_list = [checkpoint, csv_logger]
    model.fit([img_train, vel_train], labels_train, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
else:
    model = load_model(model_lst)

out = model.predict([img_test, vel_test])
np.save("./predict.npy", out)
np.save("./gt.npy", labels_test)
