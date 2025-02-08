import cv2
import pygame
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

# Initialize Pygame
pygame.init()

# Set up display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Test Poses for Pokémon")

# Initialize OpenCV camera
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Get original camera feed dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = frame_width / frame_height

# Define target size for camera feed in Pygame
target_height = screen_height // 4  # 1/4 of screen height
target_width = int(target_height * aspect_ratio)  # Maintain aspect ratio

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define the target sequence of gestures
TARGET_SEQUENCE = ["Fly", "Peace Sign", "Fist"]

# Initialize variables for sequence recognition
sequence_buffer = []  # Stores detected gestures
sequence_index = 0    # Tracks progress in target sequence
buffer_size = 30      # Buffer size

# Function to detect gestures based on hand landmarks
def detect_gesture(hand):
    """Detects predefined gestures based on hand landmark positions."""
    lmList = hand["lmList"]
    
    if not lmList:
        return "None"

    # Example: Detecting a "Fist" (all fingers curled)
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    is_fist = all(lmList[tip][1] > lmList[0][1] for tip in finger_tips)  # Tips below wrist

    # Example: Detecting a "Peace Sign" (Index & Middle extended)
    is_peace = (
        lmList[8][1] < lmList[6][1] and  # Index extended
        lmList[12][1] < lmList[10][1] and  # Middle extended
        lmList[16][1] > lmList[14][1] and  # Ring curled
        lmList[20][1] > lmList[18][1]  # Pinky curled
    )

    # Example: Detecting "Fly" (both hands open wide)
    is_fly = hand["type"] == "Left"  # Example condition, modify as needed

    if is_fist:
        return "Fist"
    elif is_peace:
        return "Peace Sign"
    elif is_fly:
        return "Fly"
    
    return "None"

# Main Loop
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

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame from BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands using cvzone
    hands_detected, img = detector.findHands(frame, draw=True)

    # Initialize detected gesture
    current_gesture = "None"

    if hands_detected:
        for hand in hands_detected:
            hand_type = hand["type"]  # "Left" or "Right"
            current_gesture = detect_gesture(hand)

            # Display hand type on screen
            bbox = hand["bbox"]
            cv2.putText(frame, f"{hand_type} Hand", (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update the sequence buffer
    if current_gesture != "None":
        sequence_buffer.append(current_gesture)

        # Keep buffer size manageable
        if len(sequence_buffer) > buffer_size:
            sequence_buffer.pop(0)

    # Check if detected sequence matches the target sequence
    if sequence_index < len(TARGET_SEQUENCE):
        target_gesture = TARGET_SEQUENCE[sequence_index]

        # Look for target gesture in buffer
        if target_gesture in sequence_buffer:
            sequence_index += 1  # Move to next gesture
            sequence_buffer.clear()  # Reset buffer
    else:
        sequence_index = 0  # Reset sequence if completed

    # Convert frame from BGR (OpenCV) to RGB (Pygame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to fit Pygame screen
    frame = cv2.resize(frame, (target_width, target_height))

    # Convert frame to Pygame surface
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)

    # Display camera feed on Pygame screen
    screen.blit(frame, (0, 0))

    # Display sequence progress
    font = pygame.font.Font(None, 36)
    text = font.render(f"Sequence Progress: {sequence_index}/{len(TARGET_SEQUENCE)}", True, (255, 255, 255))
    screen.blit(text, (screen_width // 2 - 150, screen_height - 50))

    # Update Pygame display
    pygame.display.flip()

    # Cap frame rate
    pygame.time.Clock().tick(30)

# Release resources and quit Pygame
cap.release()
pygame.quit()
