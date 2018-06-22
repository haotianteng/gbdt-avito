import keras
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Input, GlobalAveragePooling2D
import keras.backend as backend
from keras.optimizers import Adam

import tensorflow as tf

import numpy as np
import pickle

import os


def create_model( model_architecture, model_settings):
    
    if  model_architecture == 'single_fc':
        return xception_single_fc(model_settings)
    elif model_architecture == 'shallow':
        return xception_shallow(model_settings)
    # for the traning log, use all kinds structures
    else:
        raise Exception('mode_architecture: {} is not recognized'.format(model_architecture))
        

def xception_single_fc(model_settings):
    'Define a neural network with single dense layer'

    input_shape  = model_settings['input_shape']
    
    feature = Input(shape=input_shape, name='input_feature')
    X = GlobalAveragePooling2D()(feature)
    prediction = Dense(1, activation='sigmoid')(X)

    model = Model(inputs = feature, outputs = prediction)
    
    return model


def xception_shallow(model_settings):
    'Define a shallow feedforward neural network: from extracted image feature to deal probability'
    
    input_shape  = model_settings['input_shape']
    num_hidden_unit = model_settings['num_hidden_unit']
    num_hidden_layer = model_settings['num_layer'] # not used for real thing. 
    assert len(num_hidden_unit) == num_hidden_layer, "Model Setting Inconsistency: \n num_hidden_layer: {}, num_hidden_units: {}".format(num_hidden_layer, num_hidden_unit)
    
    
    feature = Input(shape=input_shape, name='input_feature')
    X = GlobalAveragePooling2D()(feature)
    for nn in num_hidden_unit:
        X = Dense(nn, activation = 'relu')(X)
    
    prediction = Dense(1, activation='sigmoid')(X)

    model = Model(inputs = feature, outputs = prediction)
    return model 
    
    
def train_utils_save(model, model_name, model_train_folder):
    'Save the trained model, architecture and weights'
    if not os.path.exists(model_train_folder):
        os.makedirs(model_train_folder)
    
    model_arch_file = os.path.join(model_train_folder, model_name + '.json')
    model_weight_file = os.path.join(model_train_folder, model_name + '.h5')

    model.save_weights(model_weight_file)
    with open(model_arch_file, 'w') as f:
        f.write(model.to_json())
        
    
    
def inference_utils_load(model_name, model_train_folder):
    'Load the trained model, architecture and weights'
    model_arch_file = os.path.join(model_train_folder, model_name + '.json')
    model_weight_file = os.path.join(model_train_folder, model_name + '.h5')
    
    if not os.path.exists(model_arch_file):
        raise Exception('Error in loading model architecture. {} not found'.format(model_arch_file))
    if not os.path.exists(model_weight_file):
        raise Exception('Error in loading model weights. {} not found'.format(model_weight_file))

    
    with open(model_arch_file, 'r') as f:
        json_string = f.read()
        model = model_from_json(json_string)
     
    model.load_weights(model_weight_file)
    return model
    
def train_utils_save_trainhistory(train_history, model_name, training_result_folder):
    'Save the training history'
    if not os.path.exists(training_result_folder):
        os.makedirs(training_result_folder)
        
    model_training_result = os.path.join(training_result_folder, model_name + '.json')
    hist_dict = train_history.history
    with open(model_training_result, 'wb') as f:
        pickle.dump(hist_dict, f)

    