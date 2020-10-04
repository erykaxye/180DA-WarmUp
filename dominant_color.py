'''
dominant color using kmeans: https://github.com/aysebilgegunduz/DominantColor
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)
clt = KMeans(n_clusters=3) #cluster number
plt.show()

while(1):
    # Take each frame
    _, frame = cap.read()

    # Draw and crop a fixed rectangle n the center of the image
    cv2.rectangle(frame,(200,150),(350,300),(255,0,0),2)
    crop = frame[200:350, 150:300]

    # Use Kmeans to find the dominant color 
    img = crop.reshape((crop.shape[0] * crop.shape[1],3)) #represent as row*column,channel number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    # Graph the histogram 
    plt.clf()
    plt.axis("off")
    plt.imshow(bar)
    plt.pause(1)

    cv2.imshow('frame',frame)
    l = cv2.waitKey(5) & 0xFF
    if l == 27:
        break

cv2.destroyAllWindows()