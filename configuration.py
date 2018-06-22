import os

data_folder = 'C:\\Juyue_Personal\\Avito_data'
training_folder = 'C:\\Juyue_Personal\\Avito_training'
train_csv_file = os.path.join(data_folder, 'train.csv', 'train.csv')
test_csv_file = os.path.join(data_folder, 'test.csv', 'test.csv')
inference_folder = os.path.join(data_folder, 'inference')
image_folder = os.path.join(data_folder, 'train_jpg_0')

## feature extraction
batch_size = 10



## different model has different model size. configure this later on.
model = 'xception'
img_size = 299
feature_size = [2, 2, 2048]

feature_folder = os.path.join(data_folder, 'img_feature', model)
log_folder = os.path.join(data_folder, 'img_feature_log', model)
# think more about it tonight.
feature_folder_inference = os.path.join(data_folder, 'img_feature_inference', model)
log_folder_inference = os.path.join(data_folder, 'img_feature_log_inference', model)

# model = 'inception_v3'
# model = 'VGG19'
# model = 'Resnet50'
# img_size = 244

## think more about this... just store the id and index? not sure.
training_data       = os.path.join(data_folder, model,'img_trainingdata')
test_data           = os.path.join(data_folder, model, 'img_testdata')


# it contains: model architecture, final weight and result. [store different weight]
training_model_folder = os.path.join(training_folder, model,'model') 
training_log_folder   = os.path.join(training_folder, model, 'log')
# it contains: training checkpoint, intermediate result, tensorboard visualization and debugging