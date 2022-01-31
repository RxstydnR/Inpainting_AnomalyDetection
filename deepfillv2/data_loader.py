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


class TrainDataLoader(tf.keras.utils.Sequence):

    def __init__(self, dir_path, batch_size=32, img_shape=(256,256), box_size=(32,32),
                 gray=False, augmentation=False, transform=None, shuffle=True):

        self.img_files = sorted(glob.glob(os.path.join(dir_path,'*.*')))
        self.img_shape = img_shape
        self.mask_size = img_shape
        self.batch_size = batch_size
        self.box_size = box_size
        
        self.gray = gray
        # self.augmentation = augmentation
        # self.transform = transform
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X_img, X_mask, X_masked, X_mask_area = self.__data_generation(img_files_temp)
        
        return [X_img, X_masked, X_mask]
                
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def _draw(self, img):

        y = np.random.randint(0, self.mask_size[0]-self.box_size[0]+1)
        x = np.random.randint(0, self.mask_size[1]-self.box_size[1]+1)
        
        # Hole size ver
        w = self.box_size[1]
        h = self.box_size[0]

        # Hole size (Random ver)
        # w = np.random.randint(self.box_min_size[1],self.box_max_size[1]+1)
        # h = np.random.randint(self.box_min_size[0],self.box_max_size[0]+1)
        
        # Mask 
        mask_area = (x,y,w,h)
        
        # Make Hole
        masked_img = img.copy()
        masked_img[y:y+h,x:x+w] = 0.0

        # Binary Mask
        bin_mask = np.zeros(masked_img.shape[0:2])
        bin_mask[y:y+h,x:x+w] = 1.0
        
        return masked_img, bin_mask, mask_area
    
    def __data_generation(self, img_files_temp):
        """ Generates data containing batch_size samples. """
        
        X_img = []
        X_masked = []
        X_mask = []
        X_mask_area = []
    
        # Generate data
        for i, img_file in enumerate(img_files_temp):

            # Read image
            if self.gray:
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
            img = cv2.resize(img, self.img_shape)
            
            masked_img, bin_mask, mask_area = self._draw(img)
            
            # Data Augmentation
            # if self.augmentation:
            #     augmented = self.transforms(image=img)
            #     img = augmented['image']
            
            img = normalize_image(img)
            masked_img = normalize_image(masked_img)
            
            X_img.append(img)
            X_masked.append(masked_img)
            X_mask.append(bin_mask)
            X_mask_area.append(mask_area)
        
        X_img = np.array(X_img).astype(np.float32)
        X_masked = np.array(X_masked).astype(np.float32)
        X_mask = np.array(X_mask).astype(np.float32)
        X_mask_area = np.array(X_mask_area).astype(np.float32)
        
        if X_img.ndim<4:
            X_img = X_img[...,np.newaxis]
        if X_masked.ndim<4:
            X_masked = X_masked[...,np.newaxis]

        return X_img,X_mask,X_masked,X_mask_area


class TestDataLoader(tf.keras.utils.Sequence):

    def __init__(self, dir_path, img_shape=(256,256), box_size=(32,32), batch_size=1, gray=False):

        self.img_files = sorted(glob.glob(os.path.join(dir_path,'*.*')))
        self.img_shape = img_shape
        self.mask_size = img_shape
        self.batch_size = 1
        self.box_size = box_size
        self.n_v_patch = img_shape[0]//box_size[0]
        self.n_h_patch = img_shape[1]//box_size[1]
        self.gray = gray
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X_img,X_mask,X_masked,X_mask_area = self.__data_generation(img_files_temp)
        
        return [X_img, X_masked, X_mask]
            
    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        self.indexes = np.arange(len(self.img_files))
    
    def _draw(self, img):
        
        # Hole size
        w = self.box_size[1]
        h = self.box_size[0]
        
        masked_imgs=[]
        bin_masks=[]
        mask_areas=[]
        
        for n_v in range(self.n_v_patch):
            for n_h in range(self.n_h_patch):
                
                # Hole location
                x = w * n_h
                y = h * n_v
                mask_area = (x,y,w,h)

                # Make Hole
                masked_img = img.copy()
                masked_img[y:y+h,x:x+w] = 0

                # Binary Mask
                bin_mask = np.zeros(masked_img.shape[0:2])
                bin_mask[y:y+h,x:x+w] = 1
                
                masked_imgs.append(masked_img)
                bin_masks.append(bin_mask)
                mask_areas.append(mask_area)
                    
        return masked_imgs, bin_masks, mask_areas
    
    def __data_generation(self, img_files_temp):
        """ Generates data containing batch_size samples. """
        
        img_file = img_files_temp[0]
        
        # Generate data
        if self.gray:
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        img = cv2.resize(img, self.img_shape)

        masked_img, bin_mask, mask_area = self._draw(img)
        
        x_img = np.array(img).astype(np.float32)
        x_masked = np.array(masked_img).astype(np.float32)
        x_mask = np.array(bin_mask).astype(np.float32)
        x_mask_area = np.array(mask_area).astype(np.float32)
        
        x_img = normalize_image(x_img)
        x_masked = normalize_image(x_masked)
        
        if x_img.ndim<3:
            x_img = x_img[...,np.newaxis]
        if x_masked.ndim<3:
            x_masked = x_masked[...,np.newaxis]

        return x_img,x_mask,x_masked,x_mask_area
    
    def _concat_patches(self,preds):
        """ Generates data containing batch_size samples. """
        
        # Hole size
        w = self.box_size[1]
        h = self.box_size[0]
        
        ret = np.zeros(preds.shape[1:],dtype=np.uint8)
        
        n=0
        for n_v in range(self.n_v_patch):
            for n_h in range(self.n_h_patch):
                
                # Hole location
                x = w * n_h
                y = h * n_v
                
                ret[y:y+h,x:x+w] = preds[n,y:y+h,x:x+w]
                n+=1
                
        return ret