import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the target sequence of gestures
TARGET_SEQUENCE = ["Thumbs Up", "Peace Sign", "Fist"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Hands object
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    # Initialize variables for sequence recognition
    sequence_buffer = []  # Stores the detected gestures
    sequence_index = 0    # Tracks progress in the target sequence
    buffer_size = 30      # Increased buffer size

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect hands
        results = hands.process(image)

        # Convert the image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initialize a variable to store the current gesture
        current_gesture = "None"

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Get landmark coordinates for gesture recognition
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Check for Thumbs Up
                if (thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and
                    thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y):
                    current_gesture = "Thumbs Up"

                # Check for Peace Sign (index and middle fingers raised, others down)
                elif (index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and
                      ring_tip.y > middle_tip.y and pinky_tip.y > middle_tip.y):
                    current_gesture = "Peace Sign"

                # Check for Fist (all fingertips below MCP joints)
                elif (thumb_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y and
                      index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
                      middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
                      ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
                      pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
                    current_gesture = "Fist"

        # Update the sequence buffer
        if current_gesture != "None":
            sequence_buffer.append(current_gesture)

            # Keep the buffer size manageable (e.g., last 30 gestures)
            if len(sequence_buffer) > buffer_size:
                sequence_buffer.pop(0)

        # Check if the sequence matches the target sequence
        if sequence_index < len(TARGET_SEQUENCE):
            target_gesture = TARGET_SEQUENCE[sequence_index]

            # Search for the target gesture in the buffer
            if target_gesture in sequence_buffer:
                sequence_index += 1  # Move to the next gesture in the sequence
                sequence_buffer = []  # Clear the buffer after a match
            else:
                # Reset if no match is found after a certain number of frames
                if len(sequence_buffer) >= buffer_size:
                    sequence_index = 0
                    sequence_buffer = []  # Clear the buffer

        # Display the current gesture and sequence progress
        cv2.putText(image, f"Current Gesture: {current_gesture}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Sequence Progress: {sequence_index}/{len(TARGET_SEQUENCE)}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Sequence Recognition', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()