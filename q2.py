# Third Party
import argparse
import cv2
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import math

def get_vanishing_points(annots):
    v_pts = []
    for prll_line_num in range(annots.shape[0]):
        # 
        points_1 = annots[prll_line_num][0]
        l1       = np.cross(np.append(points_1[:2], 1), np.append(points_1[2:], 1)).astype("float")
        l1      /= l1[-1]
        points_2 = annots[prll_line_num][1]
        l2       = np.cross(np.append(points_2[:2], 1), np.append(points_2[2:], 1)).astype("float")
        l2      /= l2[-1]
        cross_res = np.cross(l1, l2)
        v_pts.append(cross_res/cross_res[-1])
    return v_pts

def draw_vanishing_points(img, v_pts):
    vanish_img = deepcopy(img)
    ud_pad = (2*img.shape[0], 2*img.shape[0])
    lr_pad = (2*img.shape[1], 2*img.shape[1])
    vanish_img_pad = np.pad(vanish_img, (ud_pad, lr_pad, (0,0)), \
        mode="constant", constant_values=255)
    
    origin_img = np.array([lr_pad[0], ud_pad[0]])
    prev_pts = []
    for i, v_pt in enumerate(v_pts):
        pad_pt = tuple((origin_img + v_pt[:2]).astype("int"))
        
        cv2.circle(vanish_img_pad, pad_pt, 50, (0,0,255), -1)
        if len(prev_pts) > 0:
            cv2.line(vanish_img_pad, pad_pt, prev_pts[-1], (255,0,0), thickness=10)
                
        # Add points
        prev_pts.append(pad_pt)
    
    cv2.line(vanish_img_pad, prev_pts[0], prev_pts[-1], (255,0,0), thickness=10)

    cv2.imshow("Vanishing Points", cv2.resize(vanish_img_pad, (vanish_img.shape[1], vanish_img.shape[0])))
    cv2.waitKey()


def calc_K(v_pts):
    def ncr(n, r):
        return math.factorial(n) // math.factorial(r) // math.factorial(n-r)
    combinations = ncr(len(v_pts), 2)

    A = np.zeros((combinations, 4))
    idx = 0
    for i in range(len(v_pts)):
        for j in range(i+1, len(v_pts)):
            v1 = v_pts[i] / v_pts[i][-1]
            v2 = v_pts[j] / v_pts[j][-1]

            # Fill in the row of A
            A[idx, 0] = v1[0] * v2[0] + v1[1] * v2[1]
            A[idx, 1] = v2[0] * v1[2] + v1[0] * v2[2]
            A[idx, 2] = v1[2] * v2[1] + v1[1] * v2[2]
            A[idx, 3] = v1[2] * v2[2]
            idx += 1
    
    # Find flattened omega vector via SVD
    U, S, Vt = np.linalg.svd(A)
    omega_flat = Vt[np.argmin(S)]
    omega_dual = np.array([
        [omega_flat[0], 0,             omega_flat[1]],
        [0,             omega_flat[0], omega_flat[2]],
        [omega_flat[1], omega_flat[2], omega_flat[3]]
    ])

    # Use QR decomposition to obtain K
    Q, R = np.linalg.qr(omega_dual) # K^{-T}  K^{-1}
    np.linalg.cholesky(omega_dual)
    K = np.linalg.inv(R)
    return K      

def main(img_file, annot_file, output_dir):
    img    = cv2.imread(img_file)
    annots = np.load(annot_file)

    v_pts = get_vanishing_points(annots)
    draw_vanishing_points(img, v_pts)
    K = calc_K(v_pts)
    f = 0

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser("GeoViz A2:Q2 Driver", None, "")
    parser.add_argument("-i", "--img_file", default="data/q2a.png")
    parser.add_argument("-a", "--annotations", default="data/q2/q2a.npy")
    parser.add_argument("-o", "--output_dir", default="output/q2")
    args = parser.parse_args()

    # Call main
    main(args.img_file, args.annotations, args.output_dir)