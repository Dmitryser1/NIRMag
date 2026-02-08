import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

# Проверка доступности GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Вывод информации о GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU доступен!")
    print("GPU устройство:", tf.config.list_physical_devices('GPU'))
else:
    print("GPU не обнаружен. Проверьте установку CUDA/cuDNN.")

