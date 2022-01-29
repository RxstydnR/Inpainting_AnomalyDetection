import os
import warnings
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from model import GLCIC
from data_loader import TrainDataLoader

def get_available_gpus():
    """ Get the number of GPUs """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def phase1(generator,epochs=300):

    datagen = TrainDataLoader(
            dir_path=opt.dataset,
            batch_size=opt.batchSize,            
            img_shape=(opt.img_size, opt.img_size), 
            box_size=(opt.mask_size, opt.mask_size),
            phase="phase1"
    )
    
    adam = tf.keras.optimizers.Adam(lr=opt.lr)
    generator.compile(loss="mse", optimizer=adam, metrics=["mse"])
    
    history = generator.fit(
        datagen, 
        steps_per_epoch=datagen.__len__(), 
        epochs=epochs, 
        verbose=2)

    return history, generator

def phase2(model,epochs=300):

    datagen = TrainDataLoader(
            dir_path=opt.dataset,
            batch_size=opt.batchSize,            
            img_shape=(opt.img_size, opt.img_size), 
            box_size=(opt.mask_size, opt.mask_size),
            phase="phase2"
    )

    history = model.fit(
        datagen, 
        steps_per_epoch=datagen.__len__(), 
        epochs=epochs, 
        verbose=1,
        # callbacks=[EarlyStopping(monitor='loss', patience=20)],
        use_multiprocessing=False,
        workers=get_available_gpus() * 4
        )

    return history, model

def save_history(history, save_path, name="phase1"):

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(f'{save_path}/{name}.csv')

    for loss_name in history.keys():
        plt.figure(dpi=120)
        plt.plot(history[loss_name])
        plt.title(loss_name)
        plt.savefig(os.path.join(save_path,f"{name}_{loss_name}.jpg"))

    return print("saved")

def main():

    if opt.phase_learning: # Parallel learning is not supported.
        
        # glcic = GLCIC()
        # model = glcic.model
        # generator = glcic.generator

        # Phase 1
        # print("phase1",flush=True)
        # history, generator = phase1(generator,epochs=opt.nEpochs)
        # save_history(history.history, save_path=opt.SAVEPATH, name="phase1")

        # Phase 2        
        # print("phase2",flush=True)
        # generator.trainable = False 
        # model.compile(optimizer=tf.keras.optimizers.Adam(lr=opt.lr))
        # history, model = phase2(model,epochs=opt.nEpochs,compile=True)
        # save_history(history.history, save_path=opt.SAVEPATH, name="phase2")

        # Phase 3
        # print("phase3",flush=True)
        # generator.trainable = True
        # history, model = phase2(model,epochs=300)
        # save_history(history.history, save_path=SAVEPATH, name="phase3")
        raise NotImplementedError("not use phase learning")

    else:
        print("phase3",flush=True)

        with tf.distribute.MirroredStrategy().scope():
            glcic = GLCIC(
                input_shape=(opt.img_size,opt.img_size,3),
                mask_shape=(opt.mask_size,opt.mask_size,3)
            )
            model = glcic.model
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=opt.lr))

        history, model = phase2(model,epochs=opt.nEpochs)
        save_history(history.history, save_path=opt.SAVEPATH, name="phase3")

    # Save model
    generator = glcic.generator
    model.save_weights(f'{opt.SAVEPATH}/model_weights.h5')
    generator.save_weights(f'{opt.SAVEPATH}/generator_weights.h5')
    # model.save(f'{SAVEPATH}/model.h5', include_optimizer=False)
    # generator.save(f'{SAVEPATH}/generator.h5', include_optimizer=False)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='---')
    parser.add_argument('--dataset',  required=True,              help='path to data folder')
    parser.add_argument('--SAVEPATH', required=True,              help='path to save folder')
    parser.add_argument('--img_size', type=int, default=256,      help='image size')
    parser.add_argument('--mask_size',type=int, default=32,      help='image size')
    parser.add_argument('--batchSize',type=int, default=8,       help='training batch size per GPU')
    parser.add_argument('--nEpochs',  type=int, default=500,      help='number of epochs to train')
    parser.add_argument('--lr',       type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--phase_learning', action='store_true', default=False, help='phase learning')
    parser.add_argument('--GPUs',     type=str, nargs='+', default=1,        help='GPU number')
    opt = parser.parse_args()

    # display parameters
    for k, v in vars(opt).items():
        print(f'{k} = {v}')

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{','.join(opt.GPUs)}"
    os.makedirs(opt.SAVEPATH, exist_ok=True)

    main()