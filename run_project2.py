import EpipolarGeometry
import ImageFeaturesHomography
import KMeansClustering

# start task 1
task1 = ImageFeaturesHomography.ImageFeaturesHomography()
task1.start()

# start task 2
task2 = EpipolarGeometry.EpipolarGeometry()
task2.start()

# start task 3
task3 = KMeansClustering.KMeansClustering()
task3.start()
