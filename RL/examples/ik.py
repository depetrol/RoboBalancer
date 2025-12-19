import numpy as np
from ur_ikfast import ur_kinematics

ur5e = ur_kinematics.URKinematics("ur5e")
q = np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])

T = ur5e.forward(q, "matrix")
print("T base->tool0:\n", T)

pose_quat = ur5e.forward(q)
print("pose_quat:", pose_quat)


target_pose = pose_quat

solutions = ur5e.inverse(target_pose)
solutions = np.array(solutions)
print("Solution:\n", solutions)
