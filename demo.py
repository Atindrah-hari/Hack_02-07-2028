import cv2
import pygame
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()

# Set up display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pygame with Hand and Sequence Recognition")

# Initialize OpenCV camera
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Get the original dimensions of the camera feed
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = frame_width / frame_height  # Calculate aspect ratio

# Define the target size for the camera feed in Pygame
target_height = screen_height // 4  # Set target height to 1/4 of the screen height
target_width = int(target_height * aspect_ratio)  # Calculate target width to maintain aspect ratio

# Define the target sequence of gestures
TARGET_SEQUENCE = ["Thumbs Up", "Peace Sign", "Fist"]

# Initialize variables for sequence recognition
sequence_buffer = []  # Stores the detected gestures
sequence_index = 0    # Tracks progress in the target sequence
buffer_size = 30      # Increased buffer size

# Main loop
running = True
while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture frame from OpenCV
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip the frame horizontally to mirror the camera feed
    frame = cv2.flip(frame, 1)  # 1 means horizontal flip

    # Convert the frame from BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Initialize a variable to store the current gesture
    current_gesture = "None"

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,  # Draw on the original frame
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmark color
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connection color
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
        sequence_index = 0  # Reset the sequence progress

    # Convert the frame from BGR (OpenCV) to RGB (Pygame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame while preserving the aspect ratio
    frame = cv2.resize(frame, (target_width, target_height))

    # Convert the frame to a Pygame surface
    frame = np.rot90(frame)  # Rotate the frame to match Pygame's coordinate system
    frame = pygame.surfarray.make_surface(frame)

    # Clear the screen
    screen.fill((0, 0, 0))  # Fill with black

    # Blit the camera feed onto the Pygame screen
    screen.blit(frame, (0, 0))  # Position the camera feed at the top-left corner

    # Add other Pygame elements (e.g., text, shapes, etc.)
    font = pygame.font.Font(None, 36)
    text = font.render(f"Sequence Progress: {sequence_index}/{len(TARGET_SEQUENCE)}", True, (255, 255, 255))
    screen.blit(text, (screen_width // 2 - 150, screen_height - 50))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(30)

# Release the camera and quit Pygame
cap.release()
pygame.quit()