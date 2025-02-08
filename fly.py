import cv2
import pygame
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector


def distance_between_wrists(left_list,right_list):
    temp_x = (left_list[0][0] - right_list[0][0])**2
    temp_y = (left_list[0][1] - right_list[0][1])**2
    temp_z = (left_list[0][2] - right_list[0][2])**2
    return (temp_x + temp_y + temp_z)**0.5

    



def detect_fly(left_hand,right_hand):
    left_hand_list = left_hand["lmList"]
    right_hand_list = right_hand["lmList"]

    if (right_hand_list[4][0]>left_hand_list[4][0] and right_hand_list[8][0]>left_hand_list[8][0]
        and right_hand_list[12][0]>left_hand_list[12][0] and right_hand_list[16][0]>left_hand_list[16][0]
        and right_hand_list[20][0]>left_hand_list[20][0]):

        distance = distance_between_wrists(left_hand_list,right_hand_list)
        print("distance", distance)
        if (distance<200):
            return True
    return False
