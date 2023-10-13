# Third Party
import argparse
import cv2
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import math

# In House
from q1 import create_img_grid

def get_vanishing_points(annots):
    v_pts = []
    for prll_line_num in range(annots.shape[0]):
        # 
        points_1 = annots[prll_line_num][0].flatten()
        l1       = np.cross(np.append(points_1[:2], 1), np.append(points_1[2:], 1)).astype("float")
        l1      /= l1[-1]
        points_2 = annots[prll_line_num][1].flatten()
        l2       = np.cross(np.append(points_2[:2], 1), np.append(points_2[2:], 1)).astype("float")
        l2      /= l2[-1]
        cross_res = np.cross(l1, l2)
        v_pts.append(cross_res/cross_res[-1])
    return v_pts

def draw_vanishing_principal_points(img, v_pts, principal_point, output_dir):
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

    # Draw the principal point
    principal_point_adj = origin_img + np.array(principal_point)
    cv2.circle(vanish_img_pad, tuple(principal_point_adj), 10, (255,0,255), -1)

    cv2.imwrite(f"{output_dir}/vanishing_principal.png", vanish_img_pad)


def calc_K(v_pts, output_file):
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
    omega_flat = Vt[-1]

    omega = np.array([
        [omega_flat[0], 0,             omega_flat[1]],
        [0,             omega_flat[0], omega_flat[2]],
        [omega_flat[1], omega_flat[2], omega_flat[3]]
    ])
    omega /= omega[-1, -1]

    # Use cholesky to obtain K
    # Omega = K^{-T} K^{-1}
    try:
        K  = np.linalg.inv(np.linalg.cholesky(omega)).T
    except: 
        Q, R = np.linalg.qr(omega)
        K = np.linalg.inv(R)
    K /= K[-1, -1]
    np.savetxt(f"{output_file}", K)
    return K      

def main(img_file, annot_file, output_dir):
    img    = cv2.imread(img_file)
    annots = np.load(annot_file)

    v_pts = get_vanishing_points(annots)
    K     = calc_K(v_pts, f"{output_dir}/K_2a.txt")

    # Draw everything on a plot
    draw_vanishing_principal_points(img, v_pts, tuple(K[:2, -1].astype("int")), output_dir)


def cross_product_mat(vec):
    """
    Create a skew symmetric cross product matrix.

    Args:
        line_coeff (np.ndarray): [3,1] coefficients of line
    """
    return np.array([
    [0, -vec[2], vec[1]],
    [vec[2], 0, -vec[0]],
    [-vec[1], vec[0], 0]
    ])

def calc_H_correspond(src_pt, tgt_pt):
    """
    Take from Homework 1

    Args:
        src_pt (np.ndarray): Source Points
        tgt_pt (np.ndarray): Target Points

    Returns:
        np.ndarray: H matrix
    """
    # Create constraints in A matrix
    A = np.zeros((src_pt.shape[0]*2, 9))
    for i in range(src_pt.shape[0]):
        x_coeff = np.zeros((3, 9))
        x_coeff[0, :3]  = src_pt[i]
        x_coeff[1, 3:6] = src_pt[i]
        x_coeff[2, 6:]  = src_pt[i]
        x_dash_cross = cross_product_mat(tgt_pt[i])

        # Each correspondence yields 2 constraints, 3rd is a linear combination
        # of the first two
        A_idx = 2*i
        constraints = x_dash_cross.T @ x_coeff
        A[A_idx:A_idx+2] = constraints[:2]
    
    # SVD then unflatten to get H
    U, S, Vt = np.linalg.svd(A)
    # x = U[:, np.argmin(S)]
    # H = np.array([
    #     [x[0], x[1], x[2]],
    #     [x[3], x[4], x[5]],
    #     [x[6], x[7], 1.]
    # ])
    x = Vt[-1]
    H = x.reshape((3,3))
    return H

def calc_K_from_H(H_list, output_dir):
    A = np.zeros((6, 6))
    for idx, H in enumerate(H_list):
        H /= H[-1, -1]
        # Extract column
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        # Separate into pieces
        a, b, c = h1
        x, y, z = h2

        # Constraint: h_1^T \omega h_2
        A[2*idx, 0] = a*x
        A[2*idx, 1] = a*y + b*x
        A[2*idx, 2] = a*z + c*x
        A[2*idx, 3] = b*y
        A[2*idx, 4] = b*z + c*y
        A[2*idx, 5] = c*z

        # Constraint: h_1^T\omega h_1 - h_2^T \omega h_2 = 0
        A[2*idx+1, 0] = a**2 - x**2
        A[2*idx+1, 1] = 2*a*x - 2*x*y
        A[2*idx+1, 2] = 2*a*c - 2*x*z
        A[2*idx+1, 3] = b**2 - y**2
        A[2*idx+1, 4] = 2*b*c - 2*y*z
        A[2*idx+1, 5] = c**2 - z**2

    U, S, Vt = np.linalg.svd(A)
    omega_flat = Vt[-1]

    omega = np.array([
        [omega_flat[0], omega_flat[1], omega_flat[2]],
        [omega_flat[1], omega_flat[3], omega_flat[4]],
        [omega_flat[2], omega_flat[4], omega_flat[5]]
    ])

    K  = np.linalg.inv(np.linalg.cholesky(omega)).T
    K /= K[-1, -1]
    np.savetxt(f"{output_dir}/K_2b.txt", K)
    return K, omega
        

def find_H(anchor_pts, annots):
    

    H_list = []
    for square_num in range(annots.shape[0]):
        square_pts = annots[square_num]
        square_pts_homog = np.hstack((square_pts, np.ones((square_pts.shape[0], 1))))
        H = calc_H_correspond(anchor_pts, square_pts_homog)
        H_list.append(H)
    return H_list

def calc_vanishing_pt(pts):
    l_1 = np.cross(pts[0], pts[1])
    l_1 /= l_1[-1]
    l_2 = np.cross(pts[2], pts[3])
    l_2 /= l_2[-1]
    vp_1 = np.cross(l_1, l_2)
    vp_1 /= vp_1[-1]
    return vp_1

def evaluate_angles(omega, a1, a2):
    a1_h = np.hstack((a1, np.ones((a1.shape[0], 1))))
    a2_h = np.hstack((a2, np.ones((a2.shape[0], 1))))

    # Find first vanishing point
    vp_1 = calc_vanishing_pt(a1_h)

    # Find second vanishing point
    orth_a1 = np.zeros_like(a1_h)
    orth_a1[0] = a1_h[0]
    orth_a1[1] = a1_h[3]
    orth_a1[2] = a1_h[2]
    orth_a1[3] = a1_h[1]
    vp_2 = calc_vanishing_pt(orth_a1)

    # Find first vanishing line
    vl_1 = np.cross(vp_1, vp_2)
    vl_1 /= vl_1[-1]

    # Find 3rd vanishing point
    vp_3 = calc_vanishing_pt(a2_h)

    # Find 4th vanishing point
    orth_a2 = np.zeros_like(a2_h)
    orth_a2[0] = a2_h[0]
    orth_a2[1] = a2_h[3]
    orth_a2[2] = a2_h[2]
    orth_a2[3] = a2_h[1]
    vp_4 = calc_vanishing_pt(orth_a2)

    # Find the second vanishing line
    vl_2 = np.cross(vp_3, vp_4)
    vl_2 /= vl_2[-1]    

    # Create the line conic
    w_star = np.linalg.inv(omega)

    norm_1 = np.sqrt(vl_1.T @ w_star @ vl_1)
    norm_2 = np.sqrt(vl_2.T @ w_star @ vl_2)
    norm_dot_prod = (vl_1.T @ w_star @ vl_2) / (norm_1*norm_2)
    return np.arccos(norm_dot_prod) * (180/np.pi)

def visualize_annotations(img, annots, output_dir):
    imgs = {}
    for plane_num in range(annots.shape[0]):
        annot_img = deepcopy(img)
        square_pts = annots[plane_num].astype("int")
 
        cv2.fillPoly(annot_img, [square_pts], color = (255,255,255))
        imgs[f"Plane Number: {plane_num}"] = annot_img

    return imgs

def main_metric_planes(img_file, annot_file, output_dir):
    img    = cv2.imread(img_file)
    annots = np.load(annot_file)

    imgs = visualize_annotations(img, annots, output_dir)
    create_img_grid(imgs, (1,3))

    # Find homographies for each plane
    anchor_pts = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])
    H_list = find_H(anchor_pts, annots)

    # Find K
    K, omega = calc_K_from_H(H_list, output_dir)
    
    # For each pair of planes, calculate the angle
    out_file = f"{output_dir}/plane_angles.txt"
    angle_12 = evaluate_angles(omega, annots[0], annots[1])
    angle_23 = evaluate_angles(omega, annots[1], annots[2])
    angle_13 = evaluate_angles(omega, annots[0], annots[2])

    from tabulate import tabulate
    table = [["Plane 1 and 2" , angle_12],
             ["Plane 2 and 3" , angle_23],
             ["Plane 1 and 3" , angle_13]]
    tab = tabulate(table, headers=["Planes", "Angle Between (Degrees)"])
    with open(out_file, "w") as f:
        f.write(tab)


if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser("GeoViz A2:Q2 Driver", None, "")
    parser.add_argument("-i", "--img_file", default="data/q2a.png")
    parser.add_argument("-a", "--annotations", default="data/q2/q2a.npy")
    parser.add_argument("-o", "--output_dir", default="output/q2")
    args = parser.parse_args()

    # Call main
    # main(args.img_file, args.annotations, args.output_dir)

    main_metric_planes(args.img_file, args.annotations, args.output_dir)