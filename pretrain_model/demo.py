import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from HGCN.data_processing import FallDataLoader, TrainData, TestData
import HGCN.data_processing
import argparse
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from HGCN.graph import Graph
from HGCN.data_processing import FallDataLoader, TrainData, TestData
from HGCN.sgcn_lstm import Sgcn_Lstm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("inputs", r"C:\\Users\\ytkoffi\\Desktop\\sketon\\STGCN-rehab\\Data\\input.csv", "Testing")
flags.DEFINE_string("labels", r"C:\\Users\\ytkoffi\\Desktop\\sketon\\STGCN-rehab\\Data\\label.csv", "labeling")

index_Spine_Base = 0
index_Spine_Mid = 3
index_Neck = 6
index_Head = 9
index_Shoulder_Left = 12
index_Elbow_Left = 15
index_Wrist_Left = 18
index_Hand_Left = 21
index_Shoulder_Right = 24
index_Elbow_Right = 27
index_Wrist_Right = 30
index_Hand_Right = 33
index_Hip_Left = 36
index_Knee_Left = 39
index_Ankle_Left = 42
index_Foot_Left = 45
index_Hip_Right = 48
index_Knee_Right = 51
index_Ankle_Right = 54
index_Foot_Right = 57
index_Spine_Shoulder = 60
index_Tip_Left = 63
index_Thumb_Left = 66
index_Tip_Right = 69
index_Thumb_Right = 72

body_part = [
    index_Spine_Base,
    index_Spine_Mid,
    index_Neck,
    index_Head,
    index_Shoulder_Left,
    index_Elbow_Left,
    index_Wrist_Left,
    index_Hand_Left,
    index_Shoulder_Right,
    index_Elbow_Right,
    index_Wrist_Right,
    index_Hand_Right,
    index_Hip_Left,
    index_Knee_Left,
    index_Ankle_Left,
    index_Foot_Left,
    index_Hip_Right,
    index_Knee_Right,
    index_Ankle_Right,
    index_Foot_Right,
    index_Spine_Shoulder,
    index_Tip_Left,
    index_Thumb_Left,
    index_Tip_Right,
    index_Thumb_Right,
]


def Demo(inputs, labels):
    """Load the data"""
    x = pd.read_csv(inputs, header=None).iloc[:, :].values
    label = pd.read_csv(labels, header=None).iloc[:, :].values
    print(x.shape)
    print(label.shape)
   
    print("Shape of label before reshaping:", label.shape)
    # After loading the label data
    label = label.astype(int)

    batch_size = label.shape[0]
    num_timestep = 180
    num_channel = 3  
    num_joints = 25

    X = np.zeros((x.shape[0], num_joints * num_channel)).astype("float32")
    for row in range(x.shape[0]):
        counter = 0
        for parts in body_part:
            for i in range(num_channel):
                X[row, counter + i] = x[row, parts + i]
            counter += num_channel
        
    X_ = np.zeros((batch_size, num_timestep, num_joints, num_channel))
    label = np.zeros((batch_size, num_timestep, 1))

    for batch in range(min(X_.shape[0], batch_size)):
        for timestep in range(min(X_.shape[1], num_timestep)):
            for node in range(min(X_.shape[2], num_joints)):
                for channel in range(X_.shape[3]):
                    X_[batch, timestep, node, channel] = X[
                        timestep + (batch * num_timestep),
                        channel + (node * num_channel),
                    ]
                    print(x.shape)

    input_shape = (25, 3)
    X = X.reshape(-1, *input_shape)
    X = X_
    print("============================================================")
    print(X.shape)

    """Load the model and weights"""
    with open('rehabilitation.json', 'r') as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json, custom_objects={'tf': tf})
    model.load_weights('best_model.hdf5') 
    
    # Predict using the model
    y_pred = model.predict(X)
    print("Shape of y_pred before reshaping:", y_pred.shape)

    # # Apply threshold to convert y_pred to binary predictions
    # threshold = 0.5
    # y_pred_binary = (y_pred >= threshold).astype(int)

    # # Debugging print statements
    # print("Shape of y_pred before reshaping:", y_pred.shape)
    # print("Unique values in label:", np.unique(label))
    # print("True Labels (label):")
    # print(label[:, :10, :])  # Print the first 10 sequences for inspection
    # print("Predicted Labels (y_pred_binary):")
    # print(y_pred_binary[:, :10])  # Print the first 10 sequences for inspection

    # # Reshape y_pred_binary for accuracy calculation
    # y_pred_flat = y_pred_binary.reshape(-1, 1)

    # # Flatten the true labels for accuracy calculation
    # label_flat = label.reshape(-1, 1)

    # # Calculate accuracy
    # accuracy = accuracy_score(label_flat, y_pred_flat)
    # print("Accuracy:", accuracy)


# Call the Demo function
Demo(FLAGS.inputs, FLAGS.labels)




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Define the paths to your input and label CSV files
input_file_path = r"C:\\Users\\ytkoffi\\Desktop\\sketon\\STGCN-rehab\\Data\\input.csv"
label_file_path = r"C:\\Users\\ytkoffi\\Desktop\\sketon\\STGCN-rehab\\Data\\label.csv"

# Load the new data
new_data = pd.read_csv(input_file_path, header=None).iloc[:, :].values

# Create an instance of the FallDataLoader class to reuse its preprocessing steps
data_loader = FallDataLoader('')  # Replace with the actual directory

# Preprocess the new data using the same preprocessing steps
sc1 = data_loader.sc1  # Use the same StandardScaler used for training data

X_new = np.zeros(
    (new_data.shape[0], data_loader.num_joints * data_loader.num_channel)
).astype("float32")

for row in range(new_data.shape[0]):
    counter = 0
    for parts in data_loader.body_part:
        for i in range(data_loader.num_channel):
            X_new[row, counter + i] = new_data[row, parts + i]
        counter += data_loader.num_channel

# Use the same StandardScaler to scale the new data
X_new = sc1.transform(X_new)

# Now, you can use your pre-trained model to make predictions on X_new
y_pred = model.predict(X_new)
print("Shape of y_pred:", y_pred.shape)
