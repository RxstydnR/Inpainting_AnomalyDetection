import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from model import GANomaly
from data_loader import DataGenerator, denormalize_image

def save_imgs(imgs,idx,save_path):
    for i in range(len(imgs)):
        save_name = os.path.join(save_path,f"{idx}_{str(i).zfill(4)}.jpg")
        plt.imsave(save_name,imgs[i],cmap="jet",vmin=0,vmax=255)


def test(test_loader, model, save_path):

    img_dir    = os.path.join(save_path, "result", "img")
    pred_dir   = os.path.join(save_path, "result", "pred")
    anomap_dir = os.path.join(save_path, "result", "anomap")
    
    os.makedirs(img_dir)
    os.makedirs(pred_dir)
    os.makedirs(anomap_dir)

    for i in tqdm(range(test_loader.__len__())):

        # load data
        X, None = test_loader.__getitem__(i)
        X_pred = model(X).numpy()
        
        # denormalize
        # X_masked = denormalize_image(X_masked)
        # X_img = denormalize_image(X_img)
        # X_pred = denormalize_image(X_pred)

        X_anomap = np.mean(np.abs(X-X_pred),axis=-1)
        
        save_imgs([X],i,img_dir)
        save_imgs(X_pred,i,pred_dir)
        save_imgs([X_anomap],i,anomap_dir)
    
def main():

    with tf.distribute.MirroredStrategy().scope():
        
        adv_loss_weight=1.0
        cnt_loss_weight=40.0
        enc_loss_weight=1.0

        ganomaly = GANomaly(
            (opt.img_size, opt.img_size, 1),
            loss_weights=[adv_loss_weight,cnt_loss_weight,enc_loss_weight])
        model = ganomaly.build()
        model.load_weights(opt.model_weight)

    test_loader = DataGenerator(
        dir_path=opt.dataset,
        img_shape=(opt.img_size, opt.img_size), 
        box_size=(opt.mask_size, opt.mask_size),
    )

    test(test_loader, model, save_path=opt.SAVEPATH)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='---')
    parser.add_argument('--dataset', required=True, help='path to data folder')
    parser.add_argument('--SAVEPATH', required=True, help='path to save folder')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--model_weight', required=True, help='path to weight of generator')
    parser.add_argument('--GPUs', type=str, nargs='+', default=1, help='GPU number')
    opt = parser.parse_args()

    # display argparse parameters
    for k, v in vars(opt).items():
        print(f'{k} = {v}')

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{','.join(opt.GPUs)}"
    os.makedirs(opt.SAVEPATH, exist_ok=True)

    main()