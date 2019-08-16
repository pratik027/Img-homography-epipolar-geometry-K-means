import numpy as np
import cv2
UBIT = 'pratikap'; np.random.seed(sum([ord(c) for c in UBIT]))
MOUNTAIN1 = r"data\mountain1.jpg"
MOUNTAIN2 = r"data\mountain2.jpg"


class ImageFeaturesHomography:

    def task1(self, input_img, result_img):
        gray_scale_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        key_points = sift.detect(gray_scale_img, None)
        result_img_mat = cv2.drawKeypoints(input_img,key_points,color=(0,255,255), outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(result_img, result_img_mat)

    def task2(self,image_1, image_2,result_img):
        img1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        detector = cv2.xfeatures2d.SIFT_create()
        key_points1, descript1 = detector.detectAndCompute(img1, None)
        key_points2, descript2 = detector.detectAndCompute(img2, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descript1, descript2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(image_1, key_points1, image_2, key_points2, good_matches, img_matches,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(result_img, img_matches)
        return key_points1, key_points2, good_matches

    def task3(self, key_points1, key_points2, good_matches):
        src_points = np.empty((len(good_matches), 2), dtype=np.float32)
        dest_points = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            src_points[i, 0] = key_points1[good_matches[i].queryIdx].pt[0]
            src_points[i, 1] = key_points1[good_matches[i].queryIdx].pt[1]
            dest_points[i, 0] = key_points2[good_matches[i].trainIdx].pt[0]
            dest_points[i, 1] = key_points2[good_matches[i].trainIdx].pt[1]
        H, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC)
        print(H)
        return H, mask, src_points, dest_points

    def task4(self, img1, img2, key_points1, key_points2, good_matches, mask, out_file):
        matchesMask = mask.ravel().tolist()
        new_matches = []
        new_good = []
        for i in np.random.randint(0, len(matchesMask)-1, 10):
            new_matches.append(matchesMask[i])
            new_good.append(good_matches[i])

        result = cv2.drawMatches(img1, key_points1, img2, key_points2, new_good, None,matchColor=(0,255,255), matchesMask=new_matches, flags=2)
        cv2.imwrite(out_file, result)

    def task5(self,img1, img2, homography, out_file):
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        lp1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        temp = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        lp2 = cv2.perspectiveTransform(temp, homography)
        lp = np.concatenate((lp1, lp2), axis=0)

        [x_min, y_min] = np.int32(lp.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(lp.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        result = cv2.warpPerspective(img1, H_translation.dot(homography), (x_max - x_min, y_max - y_min))
        result[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img2
        cv2.imwrite(out_file, result)

    def start(self):
        img1 = cv2.imread(MOUNTAIN1)
        img2 = cv2.imread(MOUNTAIN2)

        self.task1(img1, r"data\task1_sift1.jpg")
        self.task1(img2, r"data\task1_sift2.jpg")
        print("--" * 5, "task 1.1 completed", "--" * 5)

        key_points1, key_points2, good_matches = self.task2(img1, img2, r'data\task1_matches_knn.jpg')
        print("--" * 5, "task 1.2 completed", "--" * 5)
        H, mask,_,_ =self.task3(key_points1, key_points2, good_matches)
        print("--" * 5, "task 1.3 completed", "--" * 5)
        self.task4(img1, img2, key_points1,key_points2, good_matches,mask,r"data\task1_matches.jpg")
        print("--" * 5, "task 1.4 completed", "--" * 5)
        self.task5(img1, img2,H,r"data\task1_pano.jpg")
        print("--" * 5, "task 1.5 completed", "--" * 5)


def main():
    ifh = ImageFeaturesHomography()
    ifh.start()


if __name__ == '__main__':
    main()

