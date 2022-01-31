import os
import cv2
import os
import glob
import numpy as np
import tensorflow as tf

def normalize_image(image):
    return (image / 127.5) - 1

def denormalize_image(image):
    return ((image + 1) * 127.5).astype(np.uint8)

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_files, gray=True, batch_size=32, shuffle=True, img_shape=(256,256), augmentation=False, transform=None):

        self.img_files = img_files
        self.gray = gray
        self.img_shape = img_shape
        self.augmentation = augmentation
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data. """
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X = self.__data_generation(img_files_temp)

        return X, None#,X

    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_files_temp):
        """ Generates data containing batch_size samples. """
        
        X_img = []
        # Generate data
        for i, img_file in enumerate(img_files_temp):

            # Read image
            if self.gray:
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
            img = cv2.resize(img, self.img_shape)
            
            # Data Augmentation
            #if self.augmentation:
            #    augmented = self.transforms(image=img)
            #    img = augmented['image']

            X_img.append(img)
        
        X_img = np.array(X_img).astype(np.float32)/255.

        if X_img.ndim<4:
            X_img = X_img[...,np.newaxis]

        return X_img