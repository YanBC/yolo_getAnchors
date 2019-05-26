import os
import numpy as np
import cv2 as cv
import argparse
from custom_kmeans import KMeans

##########################
# target files formats
##########################
imageExts = ['.jpg']
annoExts = ['.txt']

############################
# yolo distance functions
############################



############################
# main functions
############################

def getAnnos(dirPath, showStats=False):
    '''
    Get box sizes

    INPUT:
        dirPath <string>: path to image and annotation directory
        
        showStats <bool>: specified if show stats

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
        # convert yolo annotation to pixel annotation
        imagePath = os.path.join(dirPath, annoFile.split('.')[0] + '.jpg')
        image = cv.imread(imagePath)
        if image is None:
            print('Error in getAnnos(%s): %s not exist' % (dirPath, imagePath))
            continue
        height, width, _ = image.shape
        
        with open(os.path.join(dirPath, annoFile), 'r') as tmp:
            line = tmp.readline().strip()
            while (line != ''):
                c, x, y, w, h = line.split(' ')
                tmp_w = float(w) * width
                tmp_h = float(h) * height

                if c not in ret.keys():
                    ret[c] = [(tmp_w, tmp_h)]
                else:
                    ret[c].append((tmp_w, tmp_h))

                line = tmp.readline().strip()

    # Show stats
    if showStats:
        total = sum([len(ret[k]) for k in ret.keys()])
        print('Stats (class distributions): ')
        for k in ret.keys():
            print(k + ': ' + '%0.2f' % (len(ret[k]) / total * 100) + '%')

    return ret



def getAnchors(annos, num):
    '''
    Get anchors with kmeans

    INPUT:
        annos <dictionary>: different classes as keys, a list of boxes 
                            in that class as value
        
        num <int>: desired number of clusters

    OUTPUT:
        clusters <list>: (width, height) pairs specifying each anchors
    '''
    allboxs = []
    for k in annos.keys():
        allboxs = allboxs + annos[k]






if __name__ == '__main__':
    p = argparse.ArgumentParser(description='This script claculates a set of anchors of the given size. It uses kmeans as described in the paper. See https://arxiv.org/pdf/1612.08242.pdf for more details. ')
    p.add_argument('imageDir', help='Path to image directory. This directory should contains all the images and yolo annotations.')
    p.add_argument('clusters', help='Number of clusters. Normally this should be in the range of 3 to 9 (YOLOv3 uses 9 anchors)')
    p.add_argument('--stats', action='store_true', help='Show annotations stats')
    args = p.parse_args()

    bboxes = getAnnos(args.imageDir, args.stats)
    anchors = getAnchors(bboxes, args.clusters)