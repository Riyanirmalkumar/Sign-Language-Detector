import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load trained ISL model (expects 64x64 RGB images)
model_path = "model/isl_model_updated.h5"  # Ensure the correct path
model = tf.keras.models.load_model(model_path)

# Define class labels (A-Z)
labels = {i: chr(65 + i) for i in range(26)}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,  # Detecting  hand for better accuracy
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_label = "No Sign Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            # Convert normalized coordinates to pixel values
            x_min = int(min(x_coords) * width) - 20
            y_min = int(min(y_coords) * height) - 20
            x_max = int(max(x_coords) * width) + 20
            y_max = int(max(y_coords) * height) + 20

            # Ensure bounding box is within frame limits
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Check if cropped image is valid
            if hand_img.size == 0:
                continue

            # Resize image while preserving aspect ratio
            hand_img = cv2.resize(hand_img, (64, 64), interpolation=cv2.INTER_AREA)

            # Normalize image (convert to float and scale to [0,1])
            hand_img = hand_img.astype("float32") / 255.0

            # Expand dimensions to match model input shape
            hand_img = np.expand_dims(hand_img, axis=0)  # Shape: (1, 64, 64, 3)

            # Perform prediction
            predictions = model.predict(hand_img)
            predicted_class = np.argmax(predictions)
            predicted_label = labels.get(predicted_class, "Unknown")

            # Debugging: Show cropped hand region
            cv2.imshow("Cropped Hand", hand_img[0])  # Remove if not needed

    # Display prediction
    cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow("ISL Sign Detection", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
