# Third Party
import argparse
import cv2
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt


def calc_P(corr : np.ndarray):
    """
    Calculate P based on just the 2D-3D correspondences.

    Args:
        corr (np.ndarray): Correspondences: Row - u,v,X,Y,Z
    """
    A = np.zeros((2*corr.shape[0], 12))
    for idx, corr_row in enumerate(corr):
        # Extract out elements and form X tilde
        x,y,X,Y,Z = corr_row
        X_tilde   = np.array([X, Y, Z, 1])

        # First row of A
        A[2*idx, 4:8] = -X_tilde
        A[2*idx, 8:]  = y*X_tilde

        # Second row of A
        A[2*idx+1, :4] = X_tilde
        A[2*idx+1, 8:] = -x * X_tilde

    # Solve Ax = 0
    U, S, Vt = np.linalg.svd(A)
    p = Vt[np.argmin(S)] 
    return p.reshape((3, 4))



def project_and_view(img, P, surface_pts, bbox, out_path):
    """
    Project the points in 3D to the 2D frame given P

    Args:
        img (np.ndarray): Image of interest
        P (np.ndarray): 3x4 camera projection matrix
        surface_pts (np.ndarray): Surface points in 3D
        bbox (np.ndarray): Bounding points of the object in 3D
    """
    # Create homogenous points
    surface_pts_homog = np.hstack((surface_pts, np.ones((surface_pts.shape[0], 1))))
    bbox_pts = bbox.reshape((-1,3))
    bbox_pts_homog    = np.hstack((bbox_pts, np.ones((bbox_pts.shape[0], 1))))

    # Transform the 3D points to the image plane
    surface_pts_proj = P @ surface_pts_homog.T
    surface_pts_proj /= surface_pts_proj[-1]
    surface_pts_proj = surface_pts_proj[:2].T.astype("int")
    bbox_proj        = P @ bbox_pts_homog.T
    bbox_proj        /= bbox_proj[-1]
    bbox_proj = bbox_proj[:2].T.astype("int")

    # Create copies of the image buffer
    surface_img = deepcopy(img)
    bbox_img    = deepcopy(img)

    # Add points and save

    # For the surface/mesh
    [cv2.circle(surface_img, tuple(pt), 1, (255, 0, 0)) for pt in surface_pts_proj]
    cv2.imwrite(f"{out_path}/proj_surface.png", cv2.resize(surface_img, (surface_img.shape[1]//4, surface_img.shape[0]//4)))

    # For the bounding box
    for pt_idx in range(0, bbox_proj.shape[0]-1, 2):
        assert (pt_idx + 1) < bbox_proj.shape[0]
        cv2.line(bbox_img, tuple(bbox_proj[pt_idx]), tuple(bbox_proj[pt_idx + 1]), (0,0,255), thickness = 5)
    cv2.imwrite(f"{out_path}/proj_bbox.png", cv2.resize(bbox_img, (bbox_img.shape[1]//4, bbox_img.shape[0]//4)))

    return surface_img, bbox_img

def draw_x_faces(img, P, pts):
    """
    Draw and x on the faces that are annotated
    """
    overlay_img = deepcopy(img)
    metric_pts = pts[:, 2:]
    pts_homog  = np.hstack((metric_pts, np.ones((pts.shape[0], 1))))
    proj_pts   = (P @ pts_homog.T)
    proj_pts  /= proj_pts[-1]
    proj_pts   = proj_pts[:2].T.astype("int")

    # Find the cross points
    for idx, metric_pt in enumerate(metric_pts):
        cross_idxs = np.argwhere(np.sum((metric_pts - metric_pt) != 0, axis = 1) == 2)
        for cross_idx in cross_idxs:
            cv2.line(overlay_img, tuple(proj_pts[idx]), tuple(proj_pts[int(cross_idx)]), (255, 128, 64), thickness=5)

    cv2.imshow("Cool Image", overlay_img)
    cv2.waitKey()
    return overlay_img

def create_annotated_img(img, corr):
    ann_img = deepcopy(img)
    img_pts = corr[:, :2].astype("int")
    [cv2.circle(ann_img, tuple(pt), 10, (0, 255, 0), -1) for pt in img_pts]
    ann_img = cv2.resize(ann_img, (img.shape[1]//4, img.shape[0]//4))
    return ann_img

def create_img_grid(imgs, grid_size):
 
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 8))

    titles = list(imgs.keys())
    for i in range(axes.shape[0]):
        try:
            for j in range(axes.shape[1]):
                ax = axes[i, j]
                ax.axis("off")
                title = titles[i * axes.shape[0] + j]
                ax.set_title(title)
                ax.imshow(imgs[title])
        except:
            ax = axes[i]
            ax.axis("off")
            title = titles[i]
            ax.set_title(title)
            ax.imshow(imgs[title]) 

    # Adjust layout
    # plt.tight_layout()

    # Show the plot
    plt.show()

def main(img_file, corr_file = None, surface_pts_file = None, bbox_file = None, out_dir = "output/q1"):
    """
    Logic for simply running all parts of the code for question 1
    """
    # Load in the in the image of interest:
    img = cv2.imread(img_file)
    # item_of_interest = args.img_file.split("/")[-1].split(".")[0]
    # output_path      = f"{args.output_dir}/{item_of_interest}"

    # Load in annotations and data
    if corr_file is not None:
        corr = np.loadtxt(corr_file)
    else:
        raise ValueError("No Correspondences")

    P = calc_P(corr)
    ann_img = create_annotated_img(img, corr)

    # Once we have solved for P, project the surface points and bounding box
    if surface_pts_file is not None and bbox_file is not None:
        surface_pts = np.load(surface_pts_file)
        bbox_pts    = np.load(bbox_file)
        surface_img, bbox_img = project_and_view(img, P, surface_pts, bbox_pts, out_dir)

        # TODO: Create image grid?
        imgs = {
            "Original Image" : img,
            "Annotated Image" : ann_img,
            "Surface Points" : surface_img,
            "Bounding Box" : bbox_img
        }
        create_img_grid(imgs, grid_size=(2,2))
    else:
        x_overlay = draw_x_faces(img, P, corr)
        imgs = {
            "Original Image" : img,
            "Annotated Image" : ann_img,
            "X On Annotated Faces": x_overlay
        }
        create_img_grid(imgs, grid_size=(1,3))

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser("GeoViz A2:Q1 Driver", None, "")
    parser.add_argument("-i", "--img_file", default="data/q1/bunny.jpeg")
    parser.add_argument("-p", "--point_correspondences", default="data/q1/bunny.txt")
    parser.add_argument("-sp", "--surface_pts", default = "data/q1/bunny_pts.npy")
    parser.add_argument("-b", "--bounding_box", default = "data/q1/bunny_bd.npy")
    parser.add_argument("-o", "--output_dir", default="output/q1")
    args = parser.parse_args()

    # Call main
    main(args.img_file, args.point_correspondences, args.surface_pts, args.bounding_box, args.output_dir)