import numpy as np
import tensorflow as tf

print("NumPy version:     ", np.__version__)
print("TensorFlow version:", tf.__version__)
print("GPU available:     ", tf.config.list_physical_devices('GPU'))