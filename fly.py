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
        if (distance<200):
            return True
    return False


def detect_throw(hand,zList):
    hand_list = hand["lmList"]
    base_i_x=hand_list[8][0]
    base_i_y=hand_list[8][1]
    base_p_x=hand_list[20][0]
    base_p_y=hand_list[20][1]
    distance = ((base_i_x - base_p_x)**2 + (base_i_y - base_p_y) ** 2)**0.5
    zList.append(distance)

    delta_distance = (zList[-1] - zList[0]) // len(zList)
    print(delta_distance)



    if (len(zList)>=10):
        zList.pop(0)
    if delta_distance >=30:
        return True
    return False
  
