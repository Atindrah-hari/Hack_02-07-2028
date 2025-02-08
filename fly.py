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



    if (len(zList)>=10):
        zList.pop(0)
    if delta_distance >=30:
        return True
    return False
  


def fist_palm_tip_help(left_list,right_list):
    index_dif = abs(left_list[8][1]-right_list[8][1])
    middle_dif = abs(left_list[12][1]-right_list[12][1])
    ring_dif = abs(left_list[16][1]-right_list[16][1])
    pinky_dif = abs(left_list[20][1]-right_list[20][1])

    print("index_k_df",index_dif)
    print("middle_k_df",middle_dif)
    print("ring_k_df",index_dif)
    print("pinky_k_df",index_dif)


    return index_dif>=200 and middle_dif>=200 and ring_dif>=200 and pinky_dif>=200


def detect_fist_palm(left_hand,right_hand):
    left_hand_list = left_hand["lmList"]
    right_hand_list = right_hand["lmList"]


    index_knuckle_dif = ((left_hand_list[5][0]- right_hand_list[5][0])**2 + (left_hand_list[5][1]- right_hand_list[5][1])**2) ** 0.5 
    middle_knuckle_dif = ((left_hand_list[9][0]- right_hand_list[9][0])**2 + (left_hand_list[9][1]- right_hand_list[9][1])**2) ** 0.5 
    ring_knuckle_dif = ((left_hand_list[13][0]- right_hand_list[13][0])**2 + (left_hand_list[13][1]- right_hand_list[13][1])**2) ** 0.5 
    pinky_knuckle_dif = ((left_hand_list[17][0]- right_hand_list[17][0])**2 + (left_hand_list[17][1]- right_hand_list[17][1])**2) ** 0.5 


    




    if(index_knuckle_dif<200 and middle_knuckle_dif<200 and ring_knuckle_dif<200 and pinky_knuckle_dif<200):
        if(fist_palm_tip_help(left_hand_list,right_hand_list)):
            return True
        return False
    return False


def detect_dragonBall(left_hand, right_hand):
    left_hand_list = left_hand["lmList"]
    right_hand_list = right_hand["lmList"]
    
    dyIndex = abs(left_hand_list[8][1] - right_hand_list[8][1])
    dxIndex = abs(left_hand_list[8][0]- right_hand_list[8][0])
    
    dyMid = abs(left_hand_list[12][1] - right_hand_list[12][1])
    dxMid = abs(left_hand_list[12][0] - right_hand_list[12][0])
    
    dyRing = abs(left_hand_list[16][1] - right_hand_list[16][1])
    dxRing = abs(left_hand_list[16][0] - right_hand_list[16][0])

    dyPinky = abs(left_hand_list[20][1] - right_hand_list[20][1])
    dxPinky = abs(left_hand_list[20][0] - right_hand_list[20][0])
    
    wristDist = ((left_hand_list[0][0] - right_hand_list[0][0]) ** 2 + (left_hand_list[0][1] - right_hand_list[0][1]) ** 2) ** 0.5
    
    avgDy = (dyIndex + dyMid + dyRing + dyPinky) / 4
    avgDx = (dxRing + dxPinky + dxMid + dxIndex) / 4
    return avgDy >= 500 and avgDx <= 200 and wristDist <= 200




def detect_triangle(left_hand,right_hand):
    left_hand_list = left_hand["lmList"]
    right_hand_list = right_hand["lmList"]

    thumb_distance = ((left_hand_list[4][0]- right_hand_list[4][0])**2 + (left_hand_list[4][1]- right_hand_list[4][1])**2) ** 0.5 
    index_distance = ((left_hand_list[8][0]- right_hand_list[8][0])**2 + (left_hand_list[8][1]- right_hand_list[8][1])**2) ** 0.5

    index_avg_y =  (left_hand_list[8][1] + right_hand_list[8][1])//2
    index_avg_x =  (left_hand_list[8][0] + right_hand_list[8][0])//2
    thumb_avg_y =  (left_hand_list[4][1] + right_hand_list[4][1])//2
    thumb_avg_x =  (left_hand_list[4][1] + right_hand_list[4][1])//2


    print("index_avg_x",index_avg_x)
    print("index_avg_y",index_avg_y)
    print("thumb_avg_x",thumb_avg_x)
    print("thumb_avg_y",index_avg_y)


    return thumb_distance<150 and index_distance<150 and abs(index_avg_y -thumb_avg_y) > 350 and abs(index_avg_x -thumb_avg_x)<500