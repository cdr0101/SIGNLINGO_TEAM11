import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('model(a=66).h5')  # Ensure this matches the model saved in training

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Prepare dataset to extract unique labels
data_dir = 'assets'
labels = []

for video_name in os.listdir(data_dir):
    if video_name.endswith('.mp4'):
        labels.append(video_name.split('.')[0])

unique_labels = list(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_map.items()}

# Define function to extract keypoints from a frame
def extract_keypoints_from_frame(frame, holistic):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame_rgb)

    if result.pose_landmarks:
        pose = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in result.pose_landmarks.landmark]
    else:
        pose = np.zeros((33, 3))

    return np.array(pose)

# Function to predict the gesture from a sequence of keypoints
def predict_gesture_from_keypoints(keypoints):
    max_length_model = model.input_shape[1]  # Get the max_length from the model's input shape
    keypoints_padded = np.pad(keypoints, ((0, max_length_model - len(keypoints)), (0, 0), (0, 0)), mode='constant')
    keypoints_padded = keypoints_padded.reshape(1, max_length_model, 99, 1)  # Ensure correct shape for the model
    prediction = model.predict(keypoints_padded)
    class_idx = np.argmax(prediction)

    if class_idx in idx_to_label:
        print(f"Predicted Gesture: {idx_to_label[class_idx]}")
        return idx_to_label[class_idx]
    else:
        return "Unknown"

# Capture video from the live camera feed
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Holistic model
holistic = mp_holistic.Holistic()

keypoints_sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract keypoints from the current frame
    keypoints = extract_keypoints_from_frame(frame, holistic)
    keypoints_sequence.append(keypoints)

    # Predict gesture if we have enough keypoints
    if len(keypoints_sequence) == model.input_shape[1]:  # When sequence length matches the model's input shape
        gesture = predict_gesture_from_keypoints(np.array(keypoints_sequence))
        keypoints_sequence = []  # Clear the sequence after prediction

        # Display the predicted gesture on the frame
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()

# working for sign_language_model(a=44)
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import os

# # Load the trained model
# model = tf.keras.models.load_model('model(a=66).h5')

# # Initialize MediaPipe
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # Prepare dataset to extract unique labels
# data_dir = 'assets'
# labels = []

# for video_name in os.listdir(data_dir):
#     if video_name.endswith('.mp4'):
#         labels.append(video_name.split('.')[0])

# unique_labels = list(set(labels))
# label_map = {label: idx for idx, label in enumerate(unique_labels)}
# idx_to_label = {idx: label for label, idx in label_map.items()}

# # Define function to extract keypoints from a frame
# def extract_keypoints_from_frame(frame, holistic):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = holistic.process(frame_rgb)

#     if result.pose_landmarks:
#         pose = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in result.pose_landmarks.landmark]
#     else:
#         pose = np.zeros((33, 3))

#     return np.array(pose)

# # Function to predict the gesture from a sequence of keypoints
# def predict_gesture_from_keypoints(keypoints):
#     max_length_model = model.input_shape[1]  # Get the max_length from the model's input shape
#     keypoints_padded = np.pad(keypoints, ((0, max_length_model - len(keypoints)), (0, 0), (0, 0)), mode='constant')
#     keypoints_padded = np.expand_dims(keypoints_padded, axis=0)
#     prediction = model.predict(keypoints_padded)
#     class_idx = np.argmax(prediction)

#     if class_idx in idx_to_label:
#         print(f"Predicted Gesture: {idx_to_label[class_idx]}")
#         return idx_to_label[class_idx]
#     else:
#         return "Unknown"

# # Capture video from the live camera feed
# cap = cv2.VideoCapture(0)

# # Initialize MediaPipe Holistic model
# holistic = mp_holistic.Holistic()

# keypoints_sequence = []

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Extract keypoints from the current frame
#     keypoints = extract_keypoints_from_frame(frame, holistic)
#     keypoints_sequence.append(keypoints)

#     # Predict gesture if we have enough keypoints
#     if len(keypoints_sequence) == model.input_shape[1]:  # When sequence length matches the model's input shape
#         gesture = predict_gesture_from_keypoints(np.array(keypoints_sequence))
#         keypoints_sequence = []  # Clear the sequence after prediction

#         # Display the predicted gesture on the frame
#         cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
#     # Display the frame
#     cv2.imshow('Sign Language Recognition', frame)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# holistic.close()
