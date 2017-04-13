import tensorflow as tf
import numpy as np
import os
import time
import handle_data
from tensorflow.contrib import learn

input_train, label_train = handle_data.load_data(training)
input_eval, label_eval = handle_data.load_data(evalutation)
