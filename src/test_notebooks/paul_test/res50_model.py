import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Training data: 3512
# print(os.getcwd())
train_data = keras.utils.image_dataset_from_directory(
    'D:\\Paul_Backup\\paulj\\Pneumonia-Classification\\src\\data_split\\train',
    image_size=(256, 256), 
    batch_size=32, 
    seed = 10
)

# Validation data: 1170
val_data = keras.utils.image_dataset_from_directory(
    'D:\\Paul_Backup\\paulj\\Pneumonia-Classification\\src\\data_split\\val',
    image_size=(256, 256), 
    batch_size=32, 
    seed = 10
)

# Test data: 1174
test_data = keras.utils.image_dataset_from_directory(
    'D:\\Paul_Backup\\paulj\\Pneumonia-Classification\\src\\data_split\\test',
    image_size=(256, 256), 
    batch_size=32, 
)

def build_resnet50_model(drop_rate):
    
    # Define input shape for the model 
    inputs = keras.Input(shape = (256, 256, 3))
    # Resnet 50 basemodel 
    base_model = ResNet50(input_shape = (256, 256, 3), weights = 'imagenet', include_top = False)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x)
    x = keras.layers.Flatten()(x)
 
    x = keras.layers.Dense(256, activation = 'relu')(x)
    x = keras.layers.Dropout(drop_rate)(x)
    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def fit_model(model, train_set, validation_set):
    """Fit a model with the above stated criteria"""
    # Set patience to 5 so it doesn't take too long to fit 
    early_stopping = keras.callbacks.EarlyStopping(patience = 5)
    
    model.fit(train_set, 
              validation_data = validation_set, 
              callbacks = [early_stopping], 
              epochs = 500)
    
    return model

# set drop rate to 0.10
drop_rate = 0.10
result_dict = {}
resnet50_mod = build_resnet50_model(drop_rate)
fitted_resnet50_mod = fit_model(resnet50_mod, train_data, val_data)
result_dict[drop_rate] = fitted_resnet50_mod.evaluate(test_data)
print(result_dict)



labels = ['NORMAL', 'PNEUMONIA']
predictions = fitted_resnet50_mod.predict(test_data)

y_pred = []
y_true = []

# iterate over the dataset
for image_batch, label_batch in test_data:   # use dataset.unbatch() with repeat
   # append true labels
   y_true.append(label_batch)
   # compute predictions
   preds = fitted_resnet50_mod.predict(image_batch)
   # append predicted labels
   y_pred.append(np.where(preds > 0.5, 1,0))

print(y_true)

# convert the true and predicted labels into tensors
true_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)




cm = confusion_matrix(true_labels, predicted_labels)
print(cm)
plt.figure(figsize = (10,10))
ax = sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
ax.set(xlabel= "Prediction", ylabel = "True Value")
# ax.imshow()

classification_report(true_labels,predicted_labels)
print(classification_report(true_labels, predicted_labels, target_names = ['Normal(Class 0)','Pneumonia (Class 1)']))


# save model
# !mkdir -p saved_model
fitted_resnet50_mod.save('res50_dr12.h5')
