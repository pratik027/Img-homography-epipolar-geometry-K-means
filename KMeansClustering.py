import math
from matplotlib import pyplot as plt
import numpy as np
import cv2
UBIT = 'pratikap'; np.random.seed(sum([ord(c) for c in UBIT]))


class KMeansClustering:

    def ecludien_distance(self, p, q):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p, q)]))

    def generate_random_mu(self,shape,k):
        x = np.random.randint(0, 255, k)
        y = np.random.randint(0, 255, k)
        z = np.random.randint(0, 255, k)
        mu = []
        for x_, y_, z_ in zip(x,y,z):
            mu.append([z_,y_,z_])
        return mu

    def encode_mu(self,mu):
        return "_".join([str(x) for x in mu])

    def decode_mu(self, mu):
        return [float(a) for a in mu.split("_")]

    def task1(self, mu_list, point_list, color=['r', 'g', 'b'], out_file=r'data\\task3_iter1_a.jpg'):
        plt.clf()
        cluster_map = {}
        for mu in mu_list:
            cluster_map[self.encode_mu(mu)] = []

        for point in point_list:
            min_dis = None
            min_mu = None
            for mu in mu_list:
                dst = self.ecludien_distance(mu, point)
                if min_dis is None or dst < min_dis:
                    min_mu = mu
                    min_dis = dst
            cluster_map[self.encode_mu(min_mu)].append(point)
            plt.text(point[0]-0.11,point[1]-0.11,str(point[0])+","+str(point[1]))
        if out_file is not None:
            for i, cluster in enumerate(cluster_map):
                points = np.array(cluster_map[cluster])
                x = [float(cluster.split("_")[0])]
                y = [float(cluster.split("_")[1])]
                plt.scatter(x, y, c=color[i],edgecolors=color[i], marker="o",s=90)
                plt.text(x[0]-0.11, y[0]-0.11, str(x[0])[:3]+","+str(y[0])[:3])
                plt.scatter(points[:, 0], points[:, 1], c=color[i], edgecolors=color[i], marker="^", s=50)
            plt.savefig(out_file)
            plt.clf()
        classification_vector=[]
        for point in point_list:
            for c, mu in enumerate(cluster_map):
                if self.encode_mu(point) in [self.encode_mu(x) for x in cluster_map[mu]]:
                    classification_vector.append(c+1)
                    break
        print(classification_vector)
        return cluster_map

    def task2(self, cluster_map, color=['r', 'g', 'b'], out_file=r'data\\task3_iter1_b.jpg'):
        new_cluster = []
        for i, cluster in enumerate(cluster_map):
            x = np.average(np.array(cluster_map[cluster])[:, 0])
            y = np.average(np.array(cluster_map[cluster])[:, 1])
            new_cluster.append([x, y])
            plt.scatter([x], [y], c=color[i], edgecolors=color[i], marker="o", s=90)
            plt.text(x - 0.11, y - 0.11, str(x)[:3] + "," + str(y)[:3])
            points = np.array(cluster_map[cluster])
            for pt in points:
                plt.text(pt[0] - 0.11, pt[1] - 0.11, str(pt[0]) + "," + str(pt[1]))
            plt.scatter(points[:, 0], points[:, 1], c=color[i], edgecolors=color[i], marker="^", s=50)
        plt.savefig(out_file)
        plt.clf()
        print(new_cluster)
        return new_cluster

    def task3(self,cluster, points, color=['r', 'g', 'b'], out_file=r'data\\task3_iter2_a.jpg'):
        cluster_map = self.task1(cluster, points, color, out_file)
        self.task2(cluster_map,out_file=r'data\\task3_iter2_b.jpg')

    def cluster(self, mu_list, img):
        cluster_map = {}
        for mu in mu_list:
            cluster_map[self.encode_mu(mu)] = []

        for p,row in enumerate(img):
            for q,pixel in enumerate(row):
                min_dis = None
                min_mu = None
                for mu in mu_list:
                    dst = self.ecludien_distance(mu, pixel)
                    if min_dis is None or (dst is not None and dst < min_dis):
                        min_mu = mu
                        min_dis = dst
                cluster_map[self.encode_mu(min_mu)].append([p,q])
        return cluster_map

    def average_mu(self, img,cluster_map):
        new_cluster = {}
        new_mu = []
        for mu in cluster_map:
            r = 0
            g = 0
            b = 0
            for pixel in cluster_map[mu]:
                r += img[pixel[0]][pixel[1]][0]
                g += img[pixel[0]][pixel[1]][1]
                b += img[pixel[0]][pixel[1]][2]

            cluster_size = len(cluster_map[mu])

            if cluster_size > 0:
                r = float(r/cluster_size)
                g = float(g/cluster_size)
                b = float(b/cluster_size)
                new_mu.append([r,g,b])
                new_cluster[self.encode_mu([r,g,b])] = cluster_map[mu]
            else:
                new_mu.append(self.decode_mu(mu))
                new_cluster[mu] = cluster_map[mu]
        return new_mu, new_cluster

    def task4(self, img_name, k):
        img = cv2.imread(img_name)
        new_img = cv2.imread(img_name)
        print(img.shape)
        mu_list = self.generate_random_mu(img.shape, k)

        cnt = 50
        while cnt > 0:
            cnt -= 1
            cluster_map = self.cluster(mu_list, img)
            new_mu, cluster_map = self.average_mu(img,cluster_map)
            flag = 0
            for o_m, n_m in zip(mu_list,new_mu):
                dis = self.ecludien_distance(o_m,n_m)
                if abs(o_m[0] - n_m[0]) < 2 and abs(o_m[1] - n_m[1]) < 2 and abs(o_m[2] - n_m[2]) < 2:
                    flag += 1

            mu_list = new_mu
            if flag >= len(mu_list)/3:
                break
        print(flag,cluster_map.keys())
        for i,cluster in enumerate(cluster_map):
            clu = self.decode_mu(cluster)
            #clu = np.dot([20,20,20],i*3)
            print(clu)
            for pixel in cluster_map[cluster]:
                new_img[pixel[0]][pixel[1]] = clu

        cv2.imwrite(r'data\task3_baboon_'+str(k)+'.jpg', new_img)

    def start(self):
        point_list = [[5.9,3.2], [4.6,2.9], [6.2,2.8], [4.7,3.2], [5.5,4.2], [5.0,3.0], [4.9,3.1], [6.7,3.1], [5.1,3.8],
              [6.0,3.0]]
        MU_point = [[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]]
        cluster_map = self.task1(MU_point, point_list)
        print("--"*5,"task 3.1 completed","--"*5)
        new_mu = self.task2(cluster_map)
        print("--" * 5, "task 3.2 completed", "--"*5)
        self.task3(new_mu,point_list)
        print("--" * 5, "task 3.3 completed", "--"*5)
        # color quantization
        self.task4(r"data\baboon.jpg", 3)
        self.task4(r"data\baboon.jpg", 5)
        self.task4(r"data\baboon.jpg", 10)
        self.task4(r"data\baboon.jpg", 20)
        print("--" * 5, "task 3.4 completed", "--" * 5)


def main():
    kmc = KMeansClustering()
    kmc.start()


if __name__ == '__main__':
    main()
