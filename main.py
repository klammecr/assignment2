# In House
import q1
import q2
import q3

from q3 import annotate

# Question 1
q1.main("data/q1/bunny.jpeg", "data/q1/bunny.txt", "data/q1/bunny_pts.npy", "data/q1/bunny_bd.npy", "output/q1") # Bunny
q1.main("data/q1/cuboid.jpg", corr_file="data/q1/cuboid.txt", out_dir="output/q1") # Cuboid

# # Question 2
q2.main("data/q2a.png", "data/q2/q2a.npy", "output/q2")
q2.main_metric_planes("data/q2b.png", "data/q2/q2b.npy", "output/q2")

# Question 3
q3.main("data/q3.png", "data/q3/q3.npy", "output/q3")
q3.main("data/apple.png", None, "output/q3", annotate("data/apple.png"))
q3.main("data/box.jog", None, "output/q3", annotate("data/apple.png"))
q3.main("data/stand.jpg", None, "output/q3", annotate("data/apple.png"))