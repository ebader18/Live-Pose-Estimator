import numpy as np
import cv2 as cv2

# The rotation matrix R describes how to rotate coordinates from the coordinate system of img1 to align with the coordinate system of img2.
# The translation vector t describes the position of the origin of the coordinate system of img2 in the coordinate system of img1.
class pose_estimator:
    def __init__(self, K, img_scale=1.0, max_feature = 10000, print_messages = False):
        self.img_scale = img_scale      # scale down resolution if too high
        self.max_feature = max_feature  # Limit the number of feature to match between images
        self.K = K
        self.print_messages = print_messages
        self.kp1, self.desc1, self.kp2, self.desc2 = None, None, None, None

        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def compute_pose(self, img):
        if self.kp2 == None:    # First call of function
            self.kp2, self.desc2 = self.sift.detectAndCompute(img, None)
            self.img2 = img
            return False, None, None, None
        
        self.img1 = self.img2
        self.img2 = img
        self.kp1, self.desc1 = self.kp2, self.desc2
        self.kp2, self.desc2 = self.sift.detectAndCompute(self.img2, None)
        if self.print_messages:
            print(f'{len(self.kp1)} features detected in image #1')
            print(f'{len(self.kp2)} features detected in image #2')
        
        if len(self.desc1) == 0 or len(self.desc2) == 0:
            return False, None, None, None
        matches = self.flann.knnMatch(self.desc1[0:min(self.max_feature, len(self.desc1))], self.desc2[0:min(self.max_feature, len(self.desc2))], k=2)
        if self.print_messages:
            print(f'{len(matches)} matches found before pruning')
        pts1, pts2 = [], []
        good_matches = []
        for i,(m,n) in enumerate(matches):  # Ratio test as per Lowe's paper
            if m.distance < 0.8*n.distance:
                good_matches.append(m)
                pts2.append(self.kp2[m.trainIdx].pt)
                pts1.append(self.kp1[m.queryIdx].pt)
        if self.print_messages:
            print(f'{len(pts1)} matches found after pruning')
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        match_img = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        pts1, pts2 = np.int32(pts1), np.int32(pts2)

        if pts1.ndim != 2 or pts1.shape[1] != 2:
            return False, None, None, None
        if pts2.ndim != 2 or pts2.shape[1] != 2:
            return False, None, None, None
        F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        if self.print_messages:
            print(f'Fundamental matrix: {F}')

        if F is None or F.shape != (3, 3):
            return False, None, None, None
        E = self.K.T @ F @ self.K
        if self.print_messages:
            print(f'Essential matrix: {E}')

        points1 = np.array(pts1, dtype=np.float32)
        points2 = np.array(pts2, dtype=np.float32)
        _, R, t, mask = cv2.recoverPose(E, points1, points2, self.K)
        if self.print_messages:
            print(f'Rotation matrix: {R}')
            print(f'translation matrix: {t}')

        return True, R, t, match_img
        # Translation vector t: from img1 to img2.
        # Rotation matrix R: from img1 to img2.