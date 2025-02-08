import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)  # With drawing
    # hands = detector.findHands(img, draw=False)  # Without drawing

    # Process hand data
    if hands:
        for hand in hands:
            lmList = hand["lmList"]  # List of 21 hand landmarks
            bbox = hand["bbox"]  # Bounding box (x, y, w, h)
            center = hand["center"]  # Center of the hand
            handType = hand["type"]  # "Left" or "Right"

            # Example: Drawing a circle on the index finger tip
            index_finger_tip = lmList[8]  # Landmark 8 is the index finger tip
            cv2.circle(img, index_finger_tip[:2], 10, (0, 255, 0), -1)

    # Display output
    cv2.imshow("Hand Tracking", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
