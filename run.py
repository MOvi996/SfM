from dataloader import Dataset
from utils import * 

data = Dataset("Stage_1_Data_ver._3/stage1/box")
data.load_data()

visibility = data.visibility
indices    = np.logical_and(visibility[:,0], visibility[:,1])
pts1, pts2 = data.points[indices, 0], data.points[indices, 1]

E12, best_criterion = estimate_essential_matrix(pts1, pts2, data.K)
rotations, translations = extract_cam_pose(E12, data.K)
rot, trans = decomposeEssentialMat(E12)

for i in range(4):

    print(trans[i], translations[i])
