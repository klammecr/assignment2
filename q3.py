# Third Party
import argparse
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# In House
from q2 import calc_K
from q2 import get_vanishing_points

def homogenize_pts(annots):
    return np.array([np.hstack((annot, np.ones((annot.shape[0], 1)))) for annot in annots])

def calc_vanish_pt(p1, p2, p3, p4):
    l1       = np.cross(p1, p2).astype("float")
    l1      /= l1[-1]
    l2       = np.cross(p3, p4).astype("float")
    l2      /= l2[-1]
    cross_res = np.cross(l1, l2)
    return cross_res/cross_res[-1]

def calc_vanishing_pts(annots):
    v_pts = []
    for plane_num in range(annots.shape[0]):
        plane_pts = annots[plane_num]

        vp1 = calc_vanish_pt(plane_pts[0], plane_pts[1], plane_pts[2], plane_pts[3])
        vp2 = calc_vanish_pt(plane_pts[0], plane_pts[3], plane_pts[2], plane_pts[1])
        v_pts.append(vp1)
        v_pts.append(vp2)

    return v_pts

def compute_plane_normals(K, annots):
    """
    Compute the plane normals for each plane

    Args:
        K (np.ndarray): Intrinsics
        annots (np.ndarray): All annotations
    """
    n_list = []
    for plane_num in range(annots.shape[0]):
        plane_pts = annots[plane_num]
        v1 = calc_vanish_pt(plane_pts[0], plane_pts[1], plane_pts[3], plane_pts[2])
        d1 = np.linalg.inv(K) @ v1
        v2 = calc_vanish_pt(plane_pts[0], plane_pts[3], plane_pts[1], plane_pts[2])
        d2 = np.linalg.inv(K) @ v2
        vl = np.cross(d1, d2)
        # n  = np.linalg.inv(K) @ vl
        n_list.append(vl / np.linalg.norm(vl))
    
    return n_list

def compute_3d_pts(K, img, annots, n_list):

    # Compute rays
    pts = annots.reshape(-1, 2)
    pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))
    rays = (np.linalg.inv(K) @ pts_homog.T).T
    rays/= np.linalg.norm(rays, axis = 1).reshape(-1, 1)

    if annots.shape[0] == 5:
        # Identify point 1 as a reference point with depth 5
        ref_pt_1 = rays[1]*5
        a_p0 = -np.dot(ref_pt_1, n_list[0]) # Point 1 is on the 0th plane
        a_p1 = -np.dot(ref_pt_1, n_list[1]) # Point 1 is on the 1st plane
        a_p3 = -np.dot(ref_pt_1, n_list[3]) # Point 1 is on the 3rd plane
        a_p4 = -np.dot(ref_pt_1, n_list[4]) # Point 1 is on the 4th plane

        # We need to find a for the second plane still, lets find 3D point of p2
        t    =  -a_p0 / np.dot(rays[2], n_list[0])
        x_tilde_2 = rays[2] * t
        a_p2 = -np.dot(x_tilde_2, n_list[2])

        # Collect the a term of all planes
        a_list = [a_p0, a_p1, a_p2, a_p3, a_p4]

    else:
        ref_pt_3 = rays[3]*5
        a_p0 = -np.dot(ref_pt_3, n_list[0]) # Point 1 is on the 0th plane
        a_p1 = -np.dot(ref_pt_3, n_list[1]) # Point 1 is on the 1st plane
        a_p2 = -np.dot(ref_pt_3, n_list[2]) # Point 1 is on the 3rd plane
        a_list = [a_p0, a_p1, a_p2]
    print(a_list)

    # Intersection of the ray with the plane

    # Compute all 3D points
    colors = []
    x_tildes = []
    for i, annot in enumerate(annots):
        # Find color
        mask = cv2.fillConvexPoly(np.zeros((img.shape[0], img.shape[1])), annot.astype("int"), 255.)
        locs = np.where(mask == 255.)
        color = img[locs[0], locs[1]]
        colors.append(color)

        # Find the 3D points
        pts_xy = np.stack((locs[1], locs[0])).T
        pts_xy_homog = np.hstack((pts_xy, np.ones((pts_xy.shape[0], 1))))

        rays = (np.linalg.inv(K) @ pts_xy_homog.T).T
        rays /= np.linalg.norm(rays, axis=1).reshape(-1, 1)
        t = -a_list[i] / np.dot(rays, n_list[i])
        points_3d = t.reshape(-1, 1) * rays
        x_tildes.append(points_3d)
        
        # Debug: Look to see if the plan is covered by mask
        # miny = np.min(locs[0])
        # maxy = np.max(locs[0])
        # minx = np.min(locs[1])
        # maxx = np.max(locs[1])
        # cv2.imshow("yooo", mask)
        # cv2.waitKey()
        # colors.append(color)

    return x_tildes, colors


def annotate(impath):
    im = Image.open(impath)
    im = ImageOps.exif_transpose(im)
    im = np.array(im)

    clicks = []

    def click(event):
        x, y = event.xdata, event.ydata
        clicks.append([x, y, 1.])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(im)

    _ = fig.canvas.mpl_connect('button_press_event', click)
    plt.show()

    return clicks

def main(img_file, annot_file, out_dir, annot=None):
    img = cv2.imread(img_file)
    if annot_file is not None:
        annot = np.load(annot_file)
    else:
        annot = np.array(annot)[:, :2]
        annot = annot.reshape(-1, 4, 2)

    annot_homog = homogenize_pts(annot)
    
    # Want to make sure we arent repeating parallel lines
    if annot.shape[0] == 5:
        all_pts = annot.reshape(-1, 2)
        remapping = [0, 1, 2, 3, 4, 7, 5, 6, 10, 9, 11, 8, 12, 13, 15, 14]
        annot_rz  = all_pts[remapping].reshape(-1,2,4)
    else:
        all_pts = annot.reshape(-1, 2)
        remapping = [0, 1, 2, 3, 4, 7, 5, 6, 8, 9, 10, 11]
        annot_rz = all_pts[remapping].reshape(-1,2,4)
    vps = get_vanishing_points(annot_rz)

    # Part 1: Compute K
    K = calc_K(vps, f"{out_dir}/K_q3.txt")

    # Part 3: Compute Plane Normals
    n_list = compute_plane_normals(K, annot_homog)

    # Part 4-7: Compute rays and 3D points
    x_tildes, colors = compute_3d_pts(K, img, annot, n_list)

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data and colors
    x_tildes = np.vstack(x_tildes)
    colors = np.vstack(colors)/255.
    pcd.points = o3d.utility.Vector3dVector(x_tildes)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a visualization window and add the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GeoViz A2:Q3 Driver", None, "")
    parser.add_argument("-i", "--img_file", default="data/apple.png")
    parser.add_argument("-a", "--annotations", default=None)
    parser.add_argument("-o", "--output_dir", default="output/q3")
    args = parser.parse_args()
    
    if args.annotations is not None:
        main(args.img_file, args.annotations, args.output_dir)
    else:
        annots = annotate(args.img_file)
        main(args.img_file, None, args.output_dir, annots)