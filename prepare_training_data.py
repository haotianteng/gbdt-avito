import numpy as np
from keras.models import Sequential
from keras.utils import Sequence

import os

import pandas as pd


## you will load the deal probability all at once
class DataGenerator(Sequence):
    
    
    def __init__(self, dim, batch_size, phase, train_csv_file, data_folder_img_feature, y_shuffle = False):
        
        self.dim = dim
        
        self.validation_percent = 0.1 # 10 percent data is used for validataion
        self.train_csv_file     = train_csv_file
        self.data_folder_img_feature = data_folder_img_feature
        
        self.batch_size = batch_size
        self.phase      = phase # train, validation, inference
        
        self.item_id  = []
        self.image_id = []
        if not phase == 'inference':
            self.deal_probability = []

        self.n = 0
        self.load_item_info(phase)
        
        self.on_epoch_end()
        
        self.y_shuffle = y_shuffle

    def load_item_info(self, phase):
        'load item id, image id, and deal probability from the csv'
        df_all = pd.read_csv(self.train_csv_file)
        
        item_id = df_all['item_id'].values
        image_id = df_all['image'].values
        if not phase == 'inference':
            deal_prob = df_all['deal_probability'].values
        
        # split the data into train and validation. seed 0. To ensure training consistency
        n_data = len(df_all)
        np.random.seed(0)
        ind_perm = np.random.permutation(n_data)
        n_data_train = int(np.ceil(n_data * (1 - self.validation_percent)))
        
        if phase == 'train':
            indexes = ind_perm[:n_data_train]            
        elif phase == 'validation':
            indexes = ind_perm[n_data_train:]
        elif phase == 'inference':
            indexes = ind_perm
        else:
            raise Exception('Phase has to be one of "train", "validation", "inference"')

        # only record the corresponding items
        self.item_id = item_id[indexes]
        self.image_id = image_id[indexes]
        if not phase == 'inference':
            self.deal_prob = deal_prob[indexes]
        self.n = len(indexes)    
            
            
    def on_epoch_end(self):
        'Indexes items. Update Indexes after each epoch'
        self.indexes = np.arange(self.n)
        if not self.phase == 'inference': # no shuffling on inference case. you need the output aligned with item id
            np.random.shuffle(self.indexes)
        
        
    def __len__(self):
        'Denotes the number of batches per epoch: Make sure that a trial does not appear in one epoch more than once'
        if self.phase == 'train':
            return int(np.floor(self.n/self.batch_size))
        else:
            return int(np.ceil(self.n/self.batch_size))
    
    def __getitem__(self, ii):
        'Generate one batch of data'
        
        #indexes for this batch.
        batch_size = self.batch_size
        indexes = self.indexes[ii * batch_size: min((ii + 1) * batch_size, self.n)]
        
        if not self.phase == 'inference':
            x, y = self.__data_generation(indexes)
            return x, y
        else:
            x = self.__data_generation(indexes)
            return x
        
    def __data_generation(self, indexes):
        'Generate data containing batch_size samples' # X: (n_sampels, *dim)
        # Initialization
        X = np.empty((len(indexes), *self.dim))
        Y = np.empty((len(indexes)))
        
        # Generate data
        img_feature_folder = self.data_folder_img_feature
        
        for ii, ind in enumerate(indexes):
            
            image_id = self.image_id[ind]
            if pd.isnull(image_id):
                X[ii, ] = np.zeros((1, *self.dim))
            else:
                fn = os.path.join(img_feature_folder, image_id + '.npy')
                # check whether this file exist.
                if os.path.exists(fn):
                    X[ii, ] = np.load(fn)
                else:
                    X[ii, ] = np.zeros((1, *self.dim))
#                     print('Item id {} has image, whose id is {}, but cannot find corresponding feature file'.format(self.item_id[ind], image_id))
            if not self.phase == 'inference': 
                Y[ii]  = self.deal_prob[ind]
        
        if self.phase == 'train' and self.y_shuffle:
            np.random.shuffle(Y)
        
        if not self.phase == 'inference': 
            return X, Y
        else:
            return X
