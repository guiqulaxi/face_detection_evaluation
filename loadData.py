# -*- coding: utf-8 -*-

from __future__ import print_function

from database import *
from operator import itemgetter, attrgetter
import util
import os
import glob
import h5py
import numpy

def loadDetections(fn):
    """
        load detections from fn in the different formats
    """
    dets = []
    print ("loading ", fn)
    if os.path.splitext(fn)[1] == ".txt":
        dets = loadDetectionsPascalFormat(fn)

    elif os.path.splitext(fn)[1] == ".ramananmat":
        dets = loadDetectionsRamanan(fn)
    elif os.path.splitext(fn)[1] == ".shenmat":
        dets = loadDetectionsShen(fn)
    elif os.path.splitext(fn)[1] == ".mat":
        dets = loadDetectionsYann(fn)
    elif os.path.splitext(fn)[1] == ".csv":
        dets = loadDetectionsCSV(fn)
    else:
        print (fn)
        raise Exception("Detection file format not supported")
    return dets


def loadDetectionsYann(fn):
    f = util.loadmat(fn)
    det = []
    widths = []
    heights = []
    size = f['ids'].shape[0]
    bb = f['BB']
    for i in range(size):
        key = f['ids'][i][0][0].split('.')[0]
        conf = float(f['confidence'][i][0])
        if f.has_key("del"):
            if f["del"][i] == 1:
                continue
        x1 = float(bb[0][i])
        y1 = float(bb[1][i])
        x2 = float(bb[2][i])
        y2 = float(bb[3][i])
        det.append([key, conf, x1, y1, x2, y2])
    dets = sorted(det, key=itemgetter(1), reverse=True)
    return dets


def loadDetectionsShen(fn):
    f = util.loadmat(fn)
    det = []
    for idl, dd in enumerate(f["DetectionResults"]):
        for ff in dd[0][0]["faces"][0]:
            det.append([dd[0][0]["filename"][0][0].split(
                "\\")[-1].split(".")[0], ff[4], ff[0], ff[1], ff[0] + ff[2], ff[1] + ff[3]])
    dets = sorted(det, key=itemgetter(1), reverse=True)
    return det


def loadDetectionsCSV(fn):
    import csv
    f = open(fn, "rb")
    rd = csv.reader(f, delimiter=";")
    det = []
    rd.next()
    for idl, dd in enumerate(rd):
        det.append([dd[2].split(".")[0], 1, int(dd[7]), int(
            dd[8]), int(dd[7]) + int(dd[9]), int(dd[8]) + int(dd[10])])
    dets = sorted(det, key=itemgetter(1), reverse=True)
    return det


def loadDetectionsPascalFormat(f):
    ff = open(f)
    fdet = ff.readlines()
    det = []
    for idl, l in enumerate(fdet):
        dd = l.strip().split(" ")
        score = float(dd[1])
        dd[2] = float(dd[2])
        dd[3] = float(dd[3])
        dd[4] = float(dd[4])
        dd[5] = float(dd[5])
        w = float(dd[4]) - float(dd[2])
        h = float(dd[5]) - float(dd[3])
        # image_name, score, x, y, x2, y2 
        det.append([dd[0].split('.')[0], score, dd[2], dd[3], dd[4], dd[5]])
	# sorting all bboxes according to the descending order of their scores
    dets = sorted(det, key=itemgetter(1), reverse=True)
    return dets


def loadDetectionsRamanan(fn):
    f = util.loadmat(fn)
    ids = f['ids']
    scores = []
    if f.has_key('sc'):
        scores = f['sc']
    boxes = f['BB']
    n = len(ids)

    det = []
    for i in range(n):
        this_id = ids[i][0][0].split(".")[0]
        if not scores == []:
            this_score = scores[i][0]
        else:
            this_score = 1.0
        box = boxes[:, i]
        x1 = float(box[0])
        y1 = float(box[1])
        x2 = float(box[2])
        y2 = float(box[3])
        det.append([this_id, this_score, x1, y1, x2, y2])
        if 0:
            im = util.myimread(
                "/users/visics/mpederso/databases/afw/testimages/" + this_id + ".jpg")
            pylab.clf()
            pylab.imshow(im)
            util.box([y1, x1, y2, x2])
            pylab.draw()
            pylab.show()
            raw_input()
    dets = sorted(det, key=itemgetter(1), reverse=True)
    return dets

def loadMyDetection(detetion_path='afw.txt',mode=0):
    """
    Parameters:
    ----------
        mode==0:
            file_name
            face_num
            <x1,y1,x2,y2,score>
            <x1,y1,x2,y2,score>
        mode==1:
            file_name
            face_num
            <x1,y1,width,height,score>
            <x1,y1,width,height,score>
        etc.
        1004109301.jpg
        1
        225.928930283 241.036109924 431.988132477 515.827237129 1.000
    Returns:
    -------
    detection:[[file_name,x1,y1,x2,y2],[file_name,x1,y1,x2,y2]]
    """ 
    detection=[]
    f_detection=open(detetion_path,'r')
    while True:
        line = f_detection.readline()
        if not line:
            break
        file_name=line.strip().split('.')[0]
        face_num = int(f_detection.readline().strip())
       
        for i in xrange(face_num):
            line=f_detection.readline()
            bbox=line.strip().split()
            #major_axis_radius minor_axis_radius angle center_x center_y 1
            bbox=[float(bbox[x]) for x in range(5)]
            if(mode==1):
                bbox=util.transformBox(bbox)
            score=bbox[4]
            bbox[4]=bbox[3]
            bbox[3]=bbox[2]
            bbox[2]=bbox[1]
            bbox[1]=bbox[0]
            bbox[0]=score
            bbox.insert(0,file_name)
            detection.append(bbox)
    return detection 
def loadAFWAnnotation(anno_path='E:\\AFW\\testimages\\anno.mat'):
    '''
    Parameters:
    ----------
    Returns:
    -------
    result:list()
    [
    {'name':'1111.jpg','bbox': {np.array[x1,y1,x2,y2],np.array[x1,y1,x2,y2]}},
    {'name':'1112.jpg','bbox': {np.array[x1,y1,x2,y2],np.array[x1,y1,x2,y2]}}
    ]
    '''
    mat = h5py.File(anno_path, 'r')
    anno_list = mat[u'anno'][:]
    result=[]
    for i in range(anno_list.shape[1]):
        
        file_name=''.join([chr(x) for x in mat[anno_list[0,i]][:]])
        bboxes=[]
        for j in range(mat[anno_list[1,i]].shape[0]):
            t=mat[anno_list[1,i]][j,0]
            bbox=numpy.array(mat[t][:].flatten('F'))#(x1,y1,x2,y2)
            
            bboxes.append(bbox)
        result.append({"name":file_name,"bbox":bboxes})
    return result
