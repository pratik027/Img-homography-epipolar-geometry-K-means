import numpy as np
import cv2
from ImageFeaturesHomography import ImageFeaturesHomography
UBIT = 'pratikap'; np.random.seed(sum([ord(c) for c in UBIT]))


class EpipolarGeometry:
    def task1(self, img1,img2, output1,output2,output3):
        ifh = ImageFeaturesHomography()
        ifh.task1(img1,output1)
        ifh.task1(img2,output2)
        ifh.task2(img1, img2, output3)

    def task2(self, img_1, img_2):
        img1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        #detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
        sift = cv2.xfeatures2d.SIFT_create()
        key_points1, descript1 = sift.detectAndCompute(img1, None)
        key_points2, descript2 = sift.detectAndCompute(img2, None)
        #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        #matches = flann.knnMatch(des1, des2, k=2)
        knn_matches = flann.knnMatch(descript1, descript2, k=2)
        # -- Filter matches using the Lowe's ratio test
        points1 = []
        points2 = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                points1.append(key_points1[m.queryIdx].pt)
                points2.append(key_points2[m.trainIdx].pt)
        points1 = np.int32(points1)
        points2 = np.int32(points2)
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC)
        print(F)
        return F, mask, points1, points2

    def task3(self, F, mask,img1,img2, points1, points2):
        pts1 = points1[mask.ravel() == 1]
        pts2 = points2[mask.ravel() == 1]
        n_pts1 = []
        n_pts2 = []
        for i in np.random.randint(0, len(pts1)-1, 10):
            n_pts1.append(pts1[i])
            n_pts2.append(pts2[i])

        pts1 = np.int32(n_pts1)
        pts2 = np.int32(n_pts2)

        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = self.draw_lines(img1, img2, lines1, pts1, pts2)
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = self.draw_lines(img2, img1, lines2, pts2, pts1)
        cv2.imwrite(r'data\task2_epi_right.jpg',img5)
        cv2.imwrite(r'data\task2_epi_left.jpg',img3)

    def draw_lines(self, img_1, img_2, lines, pts1, pts2):
        r, c= cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY).shape
        img1 = cv2.cvtColor(cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        clr = np.array([10,50,255])
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            clr = np.add(clr,np.array([20,15,-20]))
            color = tuple(clr.tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    def task4(self,img1,img2):
        img_l = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
        disparity = stereo.compute(img_l, img_r)
        #import matplotlib.pyplot as plt
        cv2.imwrite(r'data\\task2_disparity.jpg',disparity)

    def start(self):
        img_left = cv2.imread(r'data\tsucuba_left.png')
        img_right = cv2.imread(r'data\tsucuba_right.png')
        self.task1(img_left,img_right,r'data\task2_sift1.jpg',r'data\task2_sift2.jpg',r'data\task2_matches_knn.jpg')
        print("--" * 5, "task 2.1 completed", "--" * 5)
        F, mask, points1, points2 = self.task2(img_left, img_right)
        print("--" * 5, "task 2.2 completed", "--" * 5)
        self.task3(F, mask,img_left,img_right, points1, points2)
        print("--" * 5, "task 2.3 completed", "--" * 5)
        img_left = cv2.imread(r'data\tsucuba_left.png')
        img_right = cv2.imread(r'data\tsucuba_right.png')
        self.task4(img_left, img_right)
        print("--" * 5, "task 2.4 completed", "--" * 5)


def main():
    eg = EpipolarGeometry()
    eg.start()


if __name__ == "__main__":
    main()
