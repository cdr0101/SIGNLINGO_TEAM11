import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Define function to extract keypoints from video
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(frame)

        pose = np.zeros((33, 3))
        face = np.zeros((468, 3))
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if result.pose_landmarks:
            pose = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.pose_landmarks.landmark])
        if result.face_landmarks:
            face = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.face_landmarks.landmark])
        if result.left_hand_landmarks:
            left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.left_hand_landmarks.landmark])
        if result.right_hand_landmarks:
            right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.right_hand_landmarks.landmark])

        # Normalize keypoints
        keypoints.append(np.concatenate([pose, face, left_hand, right_hand]))

    cap.release()
    holistic.close()

    return np.array(keypoints)

# Prepare dataset
data_dir = 'assets'
labels = []
data = []

for video_name in os.listdir(data_dir):
    if not video_name.endswith('.mp4'):
        continue  # Ignore files that are not in .mp4 format
    video_path = os.path.join(data_dir, video_name)
    keypoints = extract_keypoints(video_path)
    data.append(keypoints)
    label = video_name.split('.')[0]  # Assuming the filename is the label
    labels.append(label)

# Convert labels to categorical
unique_labels = list(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_map.items()}
labels = [label_map[label] for label in labels]
labels = to_categorical(labels)

# Pad sequences to the same length
max_length = max(len(seq) for seq in data)
data = [np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0), (0, 0)), mode='constant') for seq in data]
data = np.array(data)

# Flatten the keypoints dimensions
data = data.reshape(data.shape[0], data.shape[1], -1)

# Normalize data
data = data / np.max(data)

# Shuffle dataset
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(max_length, data.shape[2])),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(data, labels, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
model.save('model_lstm.h5')

# Load the model
model = tf.keras.models.load_model('model_lstm.h5')

# Function to predict the gesture
def predict_gesture(video_path):
    keypoints = extract_keypoints(video_path)
    max_length_model = model.input_shape[1]  # Get the max_length from the model's input shape

    # Ensure keypoints are padded to match the model's input length
    if len(keypoints) < max_length_model:
        keypoints_padded = np.pad(keypoints, ((0, max_length_model - len(keypoints)), (0, 0), (0, 0)), mode='constant')
    else:
        keypoints_padded = keypoints[:max_length_model]

    # Normalize and reshape to match the model's input shape
    keypoints_padded = keypoints_padded / np.max(keypoints_padded)
    keypoints_padded = keypoints_padded.reshape(1, max_length_model, keypoints_padded.shape[1] * keypoints_padded.shape[2])
    prediction = model.predict(keypoints_padded)
    class_idx = np.argmax(prediction)
    return unique_labels[class_idx]

# Test prediction
print(predict_gesture('assets/Right.mp4'))
print(predict_gesture('assets/Fight.mp4'))
print(predict_gesture('assets/But.mp4'))
print(predict_gesture('assets/Finish.mp4'))
print(predict_gesture('assets/F.mp4'))
print(predict_gesture('assets/World.mp4'))
print(predict_gesture('assets/You.mp4'))
print(predict_gesture('assets/C.mp4'))

# Training words
first_5_labels = labels[:5]
first_5_words = [idx_to_label[np.argmax(label)] for label in first_5_labels]

print("First 5 words in the training data:", first_5_words)


# # a=97
# import os
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization

# # Initialize MediaPipe
# mp_holistic = mp.solutions.holistic

# # Define function to extract keypoints from video
# def extract_keypoints(video_path):
#     cap = cv2.VideoCapture(video_path)
#     holistic = mp_holistic.Holistic()
#     keypoints = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = holistic.process(frame)

#         if result.pose_landmarks:
#             pose = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in result.pose_landmarks.landmark]
#         else:
#             pose = np.zeros((33, 3))

#         keypoints.append(pose)

#     cap.release()
#     holistic.close()

#     return np.array(keypoints)

# # Prepare dataset
# data_dir = 'assets'
# labels = []
# data = []

# for video_name in os.listdir(data_dir):
#     if not video_name.endswith('.mp4'):
#         continue  # Ignore files that are not in .mp4 format
#     video_path = os.path.join(data_dir, video_name)
#     keypoints = extract_keypoints(video_path)
#     data.append(keypoints)
#     label = video_name.split('.')[0]  # Assuming the filename is the label
#     labels.append(label)

# # Convert labels to categorical
# unique_labels = list(set(labels))
# label_map = {label: idx for idx, label in enumerate(unique_labels)}
# idx_to_label = {idx: label for label, idx in label_map.items()}
# labels = [label_map[label] for label in labels]
# labels = to_categorical(labels)

# # Pad sequences to the same length
# max_length = max(len(seq) for seq in data)
# data = [np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0), (0, 0)), mode='constant') for seq in data]
# data = np.array(data)

# # Reshape data for CNN input
# # Combine all dimensions except the first one (sample dimension) into one dimension for features
# data = data.reshape(data.shape[0], max_length, 99, 1)

# # Shuffle dataset
# indices = np.arange(len(data))
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]

# # Define the CNN model
# model = Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(max_length, 99, 1)),
#     MaxPooling2D(pool_size=(3, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(3, 3)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(3, 3)),
#     GlobalAveragePooling2D(),
#     BatchNormalization(),
#     Dense(512, activation='relu'),
#     Dropout(0.3),
#     Dense(512, activation='relu'),
#     Dense(len(unique_labels), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(data, labels, epochs=150, batch_size=32, validation_split=0.2)
# model.save('model2ndtry.h5')

# # Load the model
# model = tf.keras.models.load_model('model2ndtry.h5')

# # Function to predict the gesture
# def predict_gesture(video_path):
#     keypoints = extract_keypoints(video_path)
#     max_length_model = model.input_shape[1]  # Get the max_length from the model's input shape
#     keypoints_padded = np.pad(keypoints, ((0, max_length_model - len(keypoints)), (0, 0), (0, 0)), mode='constant')
#     keypoints_padded = keypoints_padded.reshape(1, max_length_model, 99, 1)
#     prediction = model.predict(keypoints_padded)
#     class_idx = np.argmax(prediction)
#     return unique_labels[class_idx]

# # Test prediction
# print(predict_gesture('assets/Right.mp4'))
# print(predict_gesture('assets/Fight.mp4'))
# print(predict_gesture('assets/But.mp4'))
# print(predict_gesture('assets/Finish.mp4'))
# print(predict_gesture('assets/F.mp4'))
# print(predict_gesture('assets/World.mp4'))
# print(predict_gesture('assets/You.mp4'))
# print(predict_gesture('assets/C.mp4'))

# # Training words
# first_5_labels = labels[:5]
# first_5_words = [idx_to_label[np.argmax(label)] for label in first_5_labels]

# print("First 5 words in the training data:", first_5_words)
