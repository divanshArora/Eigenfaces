#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 04:21:47 2018

@author: divansh
Classifier reference
http://scikit-learn.org/stable/modules/tree.html#tree
"""
import os
import numpy
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.metrics import accuracy_score

# Reference https://stackoverflow.com/questions/35723865/read-a-pgm-file-in-python
def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    v = pgmf.readline()
#    assert  v== 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

#Reference https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def face_to_vec(face_image):
    na = numpy.asarray(face_image)
    l = []
    for i in na:
        for j in i:
            l.append(j)
#    return na.flatten()
    return numpy.asarray(l)

def get_mean(faces):
    na = numpy.asarray(faces)
    return numpy.mean(na,axis=0)

def reduce(faces):
     mu = get_mean(faces)
#     print("mu dim = ",mu)
     for f in faces:
         f = f - mu        
     return faces


def get_cov(faces):
    na = numpy.cov(numpy.asarray(faces).transpose())
    print("cov matrix dim = ", na.shape)
    return na

def get_eighenvec(faces):
    faces = numpy.asarray(faces)
    faces = reduce(faces)
    cv = get_cov(faces)
    ev = numpy.linalg.eigh(cv)
    print("ev dim = ",type(ev))
    return ev

def last(n):
    return int(n[0])

def mat_to_image(mat, imgname):
    print("RFRR")

if __name__=='__main__':
    curr_dir = "/home/divansh/Desktop/Coursework/Sem6/PR_SML/Assignment2/assignment2/code/Q6/Face_data"
    angle = 25
    faces = []
    ratio = 4
    M = 45
    N = 45
    training = []
    testing = []
    testing_labels = []
    training_labels = []
    labels=[]
#    M = 192
#    N = 168
    cnt=0
    labelx=0
    dirs = get_immediate_subdirectories(curr_dir)    
    for d in dirs:
        labelx+=1
        pth = os.path.join(curr_dir,d)
        for x in os.listdir(pth):
#==============================================================================
#             if "bad" not in x and "Ambient" not in x:
#                 i = x.find('A')
#                 n = ""
#                 n += (x[i+2])+(x[i+3])+(x[i+4])
#                 n = int(n)
#                 if(n<=angle or True):
#                     with open(os.path.join(pth,x), 'rb') as f:
#                         with Image.open(f) as image:
#                             cover = resizeimage.resize_cover(image, [M, N])
#                             cover.save(x,image.format)
#                     f = open(x, 'rb')
#                     with Image.open(f) as image:
#                         mat = numpy.asarray(image.convert('L'))
#                         faces.append(face_to_vec(mat))
#             else:
#==============================================================================
                    with open(os.path.join(pth,x), 'rb') as f:
                        with Image.open(f) as image:
                            cover = resizeimage.resize_cover(image, [M, N])
                            cover.save(x,image.format)
                    f = open(x, 'rb')
                    with Image.open(f) as image:
                        mat = numpy.asarray(image.convert('L'))
                        f2v = face_to_vec(mat)
                        faces.append(f2v)
                        if cnt%2==0:
                            training.append(f2v)
                            cnt+=1
                            training_labels.append(labelx)
                        else:
                            testing.append(f2v)
                            cnt+=1
                            testing_labels.append(labelx)
                    

    w,v = get_eighenvec(faces)
    neww = []
    i=0
    for lambdax in w:
        neww.append((lambdax,i))
        i+=1

    k = 0    
    klist = []
    energylist = [0.9,0.95, 0.99]
    projection_list = []
    energy = 0.99
        
    neww = sorted(neww, key=last)
    sm=0
    tot = 0
    for i in neww:
        tot+=i[0]
    
    for j in energylist:
        k=0
        sm=0
        for i in range(len(neww)-1,0,-1):
            k+=1
            sm+=neww[i][0]
            if(sm/tot>j):
                klist.append(k)
                break
    print("k list = ", klist," for energy list = " ,energylist)

        
    
    plt.figure(1)
    mymat = (v.T)[-1]
#    print("shapewa = ",mymat.shape)
#    print("rstioa s= ", int(M),int(N))
    mymat = numpy.array(mymat).reshape(int(M),int(N))
#    print("my mat shape = ",mymat.shape)
    plt.imshow(mymat,cmap='gray')
    plt.title("at 40X40 highest e.v ")
    plt.show()
    plt.figure(2)
    mymat = (v.T)[-2]
#    print("shapewa = ",mymat.shape)
#    print("rstioa s= ", int(M),int(N))
    mymat = numpy.array(mymat).reshape(int(M),int(N))
#    print("my mat shape = ",mymat.shape)
    plt.imshow(mymat,cmap='gray')
    plt.title("at 40X40 second highest e.v ")
    plt.show()
    
    for i in range(0,len(klist)):
        k= klist[i]
        l = []
        for j in range(1,k+1):
            l.append(v.T[-j])
        projection_list.append(l)
         
    print("Classifying using 99% ")
#    for i in range(0,len(projection_list)):
#        proj = numpy.matmul(numpy.asarray(faces[0]),numpy.asarray(projection_list[i]).T)
#        print("RSHAPE",proj.shape)
#        plt.figure(i+3)
#        plt.imshow(proj,cmap='gray')
#        plt.title("image with k = "+str(klist[i]))
#        plt.show()
     















    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(training, training_labels)
    ans = clf.predict(testing)
#    wrong = 0
#    tot=0
#    for i in range(0,len(ans)):
#        if(ans[i]!=testing_labels[i]):
#            wrong+=1
#        tot+=1
    print("Decision tree accuracy on correct samples% = ",accuracy_score(testing_labels,ans)*100)
    
    training = []
    testing = []
    testing_labels = []
    training_labels = []
    labels=[]
#    M = 192
#    N = 168
    cnt=0
    labelx=0
    dirs = get_immediate_subdirectories(curr_dir)    
    for d in dirs:
        labelx+=1
        pth = os.path.join(curr_dir,d)
        for x in os.listdir(pth):
#==============================================================================
#             if "bad" not in x and "Ambient" not in x:
#                 i = x.find('A')
#                 n = ""
#                 n += (x[i+2])+(x[i+3])+(x[i+4])
#                 n = int(n)
#                 if(n<=angle or True):
#                     with open(os.path.join(pth,x), 'rb') as f:
#                         with Image.open(f) as image:
#                             cover = resizeimage.resize_cover(image, [M, N])
#                             cover.save(x,image.format)
#                     f = open(x, 'rb')
#                     with Image.open(f) as image:
#                         mat = numpy.asarray(image.convert('L'))
#                         faces.append(face_to_vec(mat))
#             else:
#==============================================================================
                    f = open(x, 'rb')
                    with Image.open(f) as image:
                        mat = numpy.asarray(image.convert('L'))
                        f2v = face_to_vec(mat)
#                        faces.append(f2v)
                        if cnt%2==0:
                            training.append(numpy.matmul(numpy.asarray(projection_list[2]), f2v))
                            cnt+=1
                            training_labels.append(labelx)
                        else:
                            testing.append(numpy.matmul(numpy.asarray(projection_list[2]), f2v))
                            cnt+=1
                            testing_labels.append(labelx)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(training, training_labels)
    ans = clf.predict(testing)
#    wrong = 0
#    tot=0
#    for i in range(0,len(ans)):
#        if(ans[i]!=testing_labels[i]):
#            wrong+=1
#        tot+=1
    print("Decision tree accuracy on 99% accuracy samples = ",accuracy_score(testing_labels,ans)*100)




