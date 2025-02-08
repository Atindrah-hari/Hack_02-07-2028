import cv2
import pygame
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from fly import *

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
frame = cv2.flip(cap.read()[1], 1)  # Read and flip in one line
aspect_ratio = frame_width / frame_height

# Define target size for camera feed in Pygame
target_height = screen_height   # 1/4 of screen height
target_width = int(target_height * aspect_ratio)  # Maintain aspect ratio

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define the target sequence of gestures
TARGET_SEQUENCE = ["Fly", "Throw", "fist_palm","dragonBall","Triangle"]

# Initialize variables for sequence recognition
sequence_buffer = []  # Stores detected gestures
sequence_index = 0    # Tracks progress in target sequence
buffer_size = 30      # Buffer size

# Function to detect gestures based on hand landmarks

# Main Loop
running = True
zL = [] # for throw calc
zR = []
left_is_throwing = False
right_is_throwing = False
# delay  = - 1
while running:
    # Handle Pygame event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture frame from OpenCV
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip frame horizontally for mirror effect
 
    # Convert frame from BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands using cvzone
    hands_detected, img = detector.findHands(frame, draw=True)

    # Initialize detected gesture
    current_gesture = "None"

    left_hand = None
    right_hand = None
    # if (delay>=0):
    #     delay -=1


    if hands_detected:
        for hand in hands_detected:
            if hand["type"] == "Left":
                left_hand = hand

                
            else:
                right_hand = hand
        
            hand_type = hand["type"]

            # Display hand type on screen
            bbox = hand["bbox"]
            cv2.putText(frame, f"{hand_type} Hand", (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    
    if (right_hand):
        # print('RIGHT',right_hand != {})
        # print('ZR>>>>>>',zR == [])
        # if (delay<=-1):
        right_is_throwing = detect_throw(right_hand,zR)
    # else:
    #     zR = []
    if(left_hand):
        # print('Left',left_hand != {})
        # if (delay<=-1):
        left_is_throwing = detect_throw(left_hand,zL)
    # else:
    #     zL = []
    is_throwing = left_is_throwing or right_is_throwing
    if is_throwing:
        # print(is_throwing)
        # delay = 50
        current_gesture = "Throw"
    if left_hand and right_hand:
        if (detect_fly(left_hand,right_hand)):
            current_gesture = "Fly"
        elif (detect_fist_palm(left_hand,right_hand)):
            current_gesture = "fist_palm"
        elif (detect_dragonBall(left_hand,right_hand)):
            current_gesture = "dragonBall"
        elif (detect_triangle(left_hand,right_hand)):
            current_gesture = "Triangle"
    

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
    pygame.time.Clock().tick(60)

# Release resources and quit Pygame
cap.release()
pygame.quit()
