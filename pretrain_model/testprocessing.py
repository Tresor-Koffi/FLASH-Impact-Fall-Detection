import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


class FallDataLoader:
    def __init__(self, dir):
        super().__init__()
        # self.num_repitation = 5
        self.num_channel = 3
        self.dir = dir
        self.body_part = self.body_parts()
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 180
        self.new_label = []
        self.train_x, self.train_y = self.import_dataset()
        self.batch_size =  self.train_y.shape[0] // self.num_timestep
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x, self.scaled_y = self.preprocessing()

    def body_parts(self):
        body_parts = [
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
            index_Ankle_Right,
            index_Spine_Shoulder,
            index_Tip_Left,
            index_Thumb_Left,
            index_Tip_Right,
            index_Thumb_Right,
        ]
        return body_parts

    def import_dataset(self):
        train_x = (
            pd.read_csv(f"./{self.dir}/Train_X.csv", header=None).iloc[:, :].values
        )
        print(train_x.shape)
        train_y = (
            pd.read_csv(f"./{self.dir}/Train_Y.csv", header=None).iloc[:, :].values
        )
        print(train_y.shape)
        return train_x, train_y

    def preprocessing(self):
        X_train = np.zeros(
            (self.train_x.shape[0], self.num_joints * self.num_channel)
        ).astype("float32")
        for row in range(self.train_x.shape[0]):
            counter = 0
            for parts in self.body_part:
                for i in range(self.num_channel):
                    X_train[row, counter + i] = self.train_x[row, parts + i]
                counter += self.num_channel

        X_train = self.sc1.fit_transform(X_train)

        X_train_ = np.zeros(
            (self.batch_size, self.num_timestep, self.num_joints, self.num_channel)
        )
        Y_train_ = np.zeros((self.batch_size, self.num_timestep))  # Y_train as 2D array
        
         # Adjust batch size for the last batch if it exceeds the data size
        last_batch_size = X_train.shape[0] % self.num_timestep
        if last_batch_size == 0:
            last_batch_size = self.num_timestep

        X_train_ = X_train_[:last_batch_size]

        Y_train_ = Y_train_[:last_batch_size]
        
        print("X_train_ shape:", X_train_.shape)
        print("Y_train_ shape:", Y_train_.shape)

        for batch in range(X_train_.shape[0]):
            for timestep in range(X_train_.shape[1]):
                for node in range(X_train_.shape[2]):
                    for channel in range(X_train_.shape[3]):
                        X_train_[batch, timestep, node, channel] = X_train[
                            timestep + (batch * self.num_timestep),
                            channel + (node * self.num_channel),
                        ]

        for batch in range(Y_train_.shape[0]):
            for timestep in range(Y_train_.shape[1]):
                Y_train_[batch, timestep] = self.train_y[batch * self.num_timestep + timestep]
            
        print("xtrain shape X_train_.shape", X_train_.shape)
        print("ytrain shape Y_train_.shape", Y_train_.shape)

        X_train = X_train_
        Y_train = Y_train_

        return X_train, Y_train


"""Load the model and weights"""
with open('rehabilitation.json', 'r') as f:
        model_json = f.read()

model = tf.keras.models.model_from_json(model_json, custom_objects={'tf': tf})
model.load_weights('best_model.hdf5') 
    
    # Predict using the model
y_pred = model.predict(X_train)
print("Shape of y_pred before reshaping:", y_pred.shape)

# Call the Demo function
def import_dataset ():
    x = (
            pd.read_csv(f"./{FLAGS.inputs}/Train_X.csv", header=None).iloc[:, :].values
        )
    print(x.shape)
    label = (
            pd.read_csv(f"./{FLAGS.labels}/Train_Y.csv", header=None).iloc[:, :].values
        )
    print(label.shape)
    return x, label