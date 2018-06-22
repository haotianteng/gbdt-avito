## construct main function.

import argparse
import os
import sys

import numpy as np
import keras.backend as backend
from keras.optimizers import Adam
# from keras.models import Model


import configuration
import prepare_training_data
import model



GlOBAL_PARAMETERS = None

def parse_argument():
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument(
        '--model_architecture',
        type = str,
        default = 'single_fc',
        help='model architecture: xception_single_fc or xception_shallow'
    )
    parser.add_argument(
        '--num_layer',
        type = int,
        default = 1,
        help='The number of hidden layers'
    )
    parser.add_argument(
        '--num_hidden_unit',
        type=str,
        default='5',
        help = 'A list of numbers determing the size of each hidden layer'
    )
    
    # train, test, or inference
    parser.add_argument(
        '--phase',
        type=str,
        default='inference',
        help='Phase can be: train, test, or inference'
    )
    
    
    # training
    parser.add_argument(
        '--learning_rate',
        type=float,
        default= 0.001,
        help='learning rate. Adam algorithm is used.'
    )
    parser.add_argument(
        '--is_shuffle',
        type=bool,
        default=0,
        help='0:No shuffle, use real data to train the algorithm. 1: Do shuffle, use the shuffled data to build baseline performance'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='batch size, the batch_size in configuration is used only for feature extraction'
    )
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=10,
        help='number of epochs for training'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        help='Which optimization method to use. Default is Adam. Default param : Beta_1=0.9, Beta_2=0.99, decay=0.01'
    )
    #
    
#     global_parameters, unparsed = parser.parse_known_args()
    
    return parser    
    
def main():
    
    # get parameter from command line.
    parser =  parse_argument()
    global_parameters, unparsed =  parser.parse_known_args()
    input_shape = configuration.feature_size 
    
    # unparse data 
    phase      = global_parameters.phase
    batch_size = global_parameters.batch_size
    
    
    model_architecture = global_parameters.model_architecture
   
    model_settings = dict()
    model_settings['input_shape']     = input_shape
    model_settings['num_hidden_unit'] = global_parameters.num_hidden_unit
    model_settings['num_layer']       = global_parameters.num_layer
    
    if phase == 'train':
        # get relevant parameters
        learning_rate = global_parameters.learning_rate
        n_epoch = global_parameters.n_epoch
        training_model_folder = configuration.training_model_folder
        training_result_folder = configuration.training_log_folder
        
        # build model.
        backend.clear_session()
        model_train =  model.create_model(model_architecture, model_settings)
            
        # compile model.
        opt  = Adam(lr=0.01, beta_1=0.9, beta_2=0.99, decay=0.01)
        model_train.compile(optimizer=opt, loss='logcosh', metrics=['mse'])
        
        # load data generator
        training_generator  = prepare_training_data.DataGenerator(input_shape, batch_size, 'train', 
                                                                 configuration.train_csv_file, 
                                                                 configuration.feature_folder, 
                                                                 y_shuffle = global_parameters.is_shuffle)
        validation_generator = prepare_training_data.DataGenerator(input_shape, batch_size, 'validation', 
                                                                 configuration.train_csv_file, 
                                                                 configuration.feature_folder, 
                                                                  y_shuffle = False)
        
        ## training
        train_history = model_train.fit_generator(generator          = training_generator,
                                                  validation_data    = validation_generator, 
                                                  epochs             = n_epoch)

        ## Saving. 
        # model and weights
        if not global_parameters.is_shuffle:
            model.train_utils_save(model_train,  model_architecture, training_model_folder)
        # training result. name = model_architecture + 'baseline' + '0'/'1'
        model.train_utils_save_trainhistory(train_history, 
                                            model_architecture + '_baseline_' + str(global_parameters.is_shuffle), 
                                            training_result_folder)  
        
        
    elif phase == 'test':
        # do we even have a testing data. maybe not.
        print('testing')
    
    elif phase == 'inference':
        
        # load model
        training_model_folder = configuration.training_model_folder
        inference_storage_folder = configuration.inference_folder
        
        model_train = model.inference_utils_load(model_architecture, training_model_folder)
        
        # load inference generator.
        inference_generator  = prepare_training_data.DataGenerator(input_shape, batch_size, 'inference', 
                                                 configuration.test_csv_file, 
                                                 configuration.feature_folder_inference)
        # predict
        prediction = model_train.predict_generator(inference_generator)
        item_id = inference_generator.item_id

        # save
        inference_save_prediction(item_id, prediction, inference_storage_folder)
        
    else:
        raise Exception('Phase: {} is not recognized. It has to be one of {train, test, inference}'.format(phase))

        
def inference_save_prediction(item_id, prediction, inference_folder):
    if not os.path.exists(inference_folder):
        os.makedirs(inference_folder)
    fn = os.path.join(configuration.inference_result, 'prediction')
    result = np.vstack((item_id, np.squeeze(prediction)))
    np.save(fn, result)
    
    
if __name__ == '__main__':
    
    main()
    