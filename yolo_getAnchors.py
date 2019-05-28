import os
import numpy as np
import cv2 as cv
import argparse
import sys
from custom_kmeans import KMeans

##########################
# target files formats
##########################
imageExts = ['.jpg']
annoExts = ['.txt']



############################
# yolo distance functions
############################

def IOU(box1, box2):
    w1, h1 = box1
    w2, h2 = box2
    if w1>=w2 and h1>=h2:
        iou = w2*h2/(w1*h1)
    elif w1>=w2 and h1<=h2:
        iou = w2*h1/(w2*h2 + (w1-w2)*h1)
    elif w1<=w2 and h1>=h2:
        iou = w1*h2/(w2*h2 + w1*(h1-h2))
    else:
        iou = (w1*h1)/(w2*h2)
    return iou


def yolo_distance(x, centroid):
    return 1 - IOU(x, centroid)


def yolo_findCentroid(x_array):
    assert x_array.shape[1] == 2
    return np.sum(x_array, axis=0) / x_array.shape[0]


def simi(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        similarity = IOU(x, centroid)
        similarities.append(similarity)
    return np.array(similarities) 


def avg_IOU(X, centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum += max(simi(X[i],centroids)) 
    return sum/n



############################
# main functions
############################

def getAnnos(dirPath, image_shape=None, showStats=False):
    '''
    Get box sizes

    INPUT:
        dirPath <string>: path to image and annotation directory
        
        image_shape <pairs of int>: yolo network input image shape, (width, height);
                                    specified if images are shrunk before feeding to 
                                    yolo network

        showStats <bool>: whether to show stats or not

    OUTPUT:
        ret <dictionary>: different classes as keys, a list of boxes 
                            in that class as value

    NOTES:
        Directory should look like:
        |
        ├── frame126.jpg
        ├── frame126.txt
        ├── frame127.jpg
        ├── frame127.txt
        ├── frame128.jpg
        ├── frame128.txt
        ├── frame129.jpg
        ├── frame129.txt
        ├── frame130.jpg
        ├── frame130.txt
        ├── frame131.jpg
        ├── frame131.txt
        ├── frame132.jpg
        ├── frame132.txt
        ├── frame133.jpg
    '''
    files = os.listdir(dirPath)
    images = [f for f in files if (imageExts[0] in f)]
    annoFiles = [f for f in files if (annoExts[0] in f)]

    ret = dict()
    for annoFile in annoFiles:

        # get image informations
        imagePath = os.path.join(dirPath, annoFile.split('.')[0] + '.jpg')
        image = cv.imread(imagePath)
        if image is None:
            print('Error in getAnnos(%s): %s not exist' % (dirPath, imagePath))
            continue
        height, width, _ = image.shape

        # if image_shape is specified, shrink image width and height
        # else, do nothing
        if image_shape:
            scale_height = height / image_shape[1]
            scale_width = width / image_shape[0]
        else:
            scale_height = 1
            scale_width = 1
        
        # read annotation
        with open(os.path.join(dirPath, annoFile), 'r') as tmp:
            line = tmp.readline().strip()
            while (line != ''):
                c, x, y, w, h = line.split(' ')
                tmp_w = float(w) * width / scale_width
                tmp_h = float(h) * height / scale_height

                if c not in ret.keys():
                    ret[c] = [(tmp_w, tmp_h)]
                else:
                    ret[c].append((tmp_w, tmp_h))

                line = tmp.readline().strip()

    # Show stats
    if showStats:
        total = sum([len(ret[k]) for k in ret.keys()])
        print('Total annotations: %d' % total)
        print('Stats (class distributions): ')
        for k in ret.keys():
            print(k + ': ' + '%0.2f' % (len(ret[k]) / total * 100) + '%')

    return ret



def getAnchors(X, k):
    '''
    Get anchors with kmeans

    INPUT:
        X <dictionary>: different classes as keys, a list of boxes 
                            in that class as value
        
        k <int>: desired number of clusters

    OUTPUT:
        clusters <list>: (width, height) pairs specifying each anchors
    '''
    kmeans = KMeans(k, 10, yolo_distance, yolo_findCentroid, 5000)
    kmeans.fit(X)
    return kmeans.inertia_, kmeans.cluster_centers_





if __name__ == '__main__':
    p = argparse.ArgumentParser(description='This script claculates a set of anchors of the given size. It uses kmeans as described in the paper. See https://arxiv.org/pdf/1612.08242.pdf for more details.')
    p.add_argument('imageDir', help='Path to image directory. This directory should contains all the images and yolo annotations.')
    p.add_argument('clusters', help='Number of clusters. Normally this should be in the range of 3 to 9 (YOLOv3 uses 9 anchors)')
    p.add_argument('--stats', action='store_true', help='Show annotations stats')
    p.add_argument('--width', help='yolo network shrunk input image width; you have to specify both -w and -h to use this option; DEFAULT: No shrinking')
    p.add_argument('--height', help='yolo network shrunk input image height; you have to specify both -w and -h to use this option; DEFAULT: No shrinking')
    args = p.parse_args()

    # check args.width and args.height options
    if (args.width and not args.height) or (not args.width and args.height):
        print("\nYou should use both --width and --height to make it effective")
        print("See <python3 %s -h> for more informations" % sys.argv[0])
        exit(-1)

    # Get all annotated bounding boxes
    if args.width and args.height:
        bboxes = getAnnos(args.imageDir, image_shape=(int(args.width), int(args.height)), showStats=args.stats)
    else:
        bboxes = getAnnos(args.imageDir, showStats=args.stats)
    allboxs = []
    for key in bboxes.keys():
        allboxs = allboxs + bboxes[key]
    X = np.array(allboxs)
    assert X.shape == (len(allboxs), 2)

    # Get anchors with kmeans
    kmean_loss, anchors = getAnchors(X, int(args.clusters))

    # Calculate avarage IOU
    n_c, _ = anchors.shape
    centroids = []
    for i in range(n_c):
        centroids.append((anchors[i,0], anchors[i,1]))
    avg = avg_IOU(X, centroids)

    # Print stats
    print('Sum of distances for each point to its centroid: %0.4f' % kmean_loss)
    print('Avarage IOU: %0.4f' % avg)
    print('List of Anchors: ')
    tmp = np.array(centroids)
    tmp = tmp[(tmp[:,0] * tmp[:,1]).argsort()]
    for i in range(len(tmp)):
        print(np.array(tmp[i]).astype(int))
