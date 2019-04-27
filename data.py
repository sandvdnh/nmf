import numpy as np
import re
import os
import glob
import math


def synthetic(config):
    '''
    returns random positive matrices v, w, h such that v = w * h and:
    - v has shape (n, m)
    - w has shape (n, r)
    - h has shape (r, m)
    '''
    n = config['n']
    m = config['m']
    r = config['r']
    w = np.abs(np.random.normal(size=(n, r), scale=2, loc=3))
    h = np.abs(np.random.normal(size=(r, m), scale=2, loc=3))
    v = np.matmul(w, h)
    ground_truth = (w, h)
    v = np.abs(np.random.normal(size=(n, m), scale=2, loc=5))
    return v, ground_truth


def face(config):
    '''
    loads faces in ./data/faces/
    '''
    path = os.path.join(config['path'], 'face')
    path = os.path.join(path, '*.pgm')
    wav_files = glob.glob(path)
    X = np.zeros((361, len(wav_files)))
    for i, file_ in enumerate(wav_files):
        a = pgmread(file_).reshape((361,))
        X[:, i] = a.copy()
    return X, -1


def pgmread(filename):
    """  This function reads Portable GrayMap (PGM) image files and returns
    a numpy array. Image needs to have P2 or P5 header number.
    Line1 : MagicNum
    Line2 : Width Height
    Line3 : Max Gray level
    Lines starting with # are ignored """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))
    #f = open(filename,'r')
    ## Read header information
    #count = 0
    #while count < 3:
    #    line = f.readline()
    #    if line[0] == '#': # Ignore comments
    #        continue
    #    count = count + 1
    #    if count == 1: # Magic num info
    #        magicNum = line.strip()
    #        if magicNum != 'P2' and magicNum != 'P5':
    #            f.close()
    #            #print 'Not a valid PGM file'
    #            exit()
    #    elif count == 2: # Width and Height
    #        [width, height] = (line.strip()).split()
    #        width = int(width)
    #        height = int(height)
    #    elif count == 3: # Max gray level
    #        maxVal = int(line.strip())
    ## Read pixels information
    #img = []
    #buf = f.read()
    #elem = buf.split()
    #if len(elem) != width*height:
    #    #print 'Error in number of pixels'
    #    exit()
    #for i in range(height):
    #    tmpList = []
    #    for j in range(width):
    #        tmpList.append(elem[i*width+j])
    #    img.append(tmpList)
    #return (np.array(img), width, height)

def get_data(config):
    '''
    constructs data matrix based on config
    '''
    if config['dataset'] == 'synthetic':
        X, ground_truth = synthetic(config)
    if config['dataset'] == 'face':
        X, ground_truth = face(config)
    return X, ground_truth
