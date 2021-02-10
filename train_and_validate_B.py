#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd


# In[2]:


classes = os.listdir("./dataset_B/")


# In[3]:


img_in = Input((320,240,3))
model = DenseNet121(include_top= False, 
                weights='imagenet',     
                input_tensor= img_in, 
                input_shape= (320,240,3),
                pooling ='avg')
x = model.output 
predictions = Dense(len(classes[:10]), activation="softmax", name="predictions")(x)    # fully connected layer for predict class 
model = Model(inputs=img_in, outputs=predictions)


# In[4]:


test_datagen = ImageDataGenerator()


# In[5]:


train_generator = test_datagen.flow_from_directory('./dataset_B/',classes=classes[:10],target_size = (320, 240),color_mode = 'rgb')


# In[ ]:


optimizer = Adam(lr=0.01)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit_generator(generator=train_generator,
                    use_multiprocessing=True,
                    epochs = 30,
                    workers = 4)


# In[ ]:


model.save("model_B.h5")

