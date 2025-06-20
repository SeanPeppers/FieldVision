import argparse
import numpy as np
import cv2
import glob
import os

'''
example syntax

python calibration.py -image_path /data/e/stand_counts/stand_count_dataset/30_2019/raw/


'''


parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-image_path', type=str,  help="paths to directory")
parser.add_argument('-save_path', type=str,  help="paths to save directory")
parser.add_argument('-xxx', '--xxx', dest='xxx', default = 0, type=float,  help='x rad calib')
parser.add_argument('-yyy', '--yyy', dest='yyy', default = 0, type=float,  help='y rad calib')
parser.add_argument('-zzz', '--zzz', dest='zzz', default = 0, type=float,  help='z rad calib')

args = parser.parse_args()

paths = args.image_path + '/*.png'
calibrated_paths = args.save_path  # + '/calibrated/'
#homography_matrix_path = args.image_path + '/homography_matrix/'

images = sorted(glob.glob(paths))


if not os.path.exists(calibrated_paths):
   os.makedirs(calibrated_paths)


#if not os.path.exists(homography_matrix_path):
#   os.makedirs(homography_matrix_path)

counter = 0


#img_init = cv2.imread(images[0])
#h,  w = img_init.shape[:2]
'''
mtx = np.array([ [2359.79036, 0, 2003.35175],[0 , 2359.30434, 1034.37100], [0, 0, 1 ] ]  )
dist = np.array([ -0.00744 ,  -0.01876 ,  -0.00308 ,  -0.00200,  0.00000 ])
'''

## trial and error gimbal calibration
def roty(theta):

  ct = np.cos(theta)
  st = np.sin(theta)

  R=np.array([[ct, 0, st],[0,1,0],[-st, 0, ct]])

  return R

def rotx(t):
  ct = np.cos(t)
  st = np.sin(t)

  R=np.array([[1,0,0],[0, ct, -st],[0, st, ct]])

  return R

def rotz(t):
  ct = np.cos(t)
  st = np.sin(t)
  
  R=np.array([[ct,-st,0],[st, ct, 0],[0, 0, 1]])

  return R

print('test')


#Ry = np.dot(roty(-0.053),rotx(0.12))

if args.xxx != 0 and args.zzz == 0 and args.yyy == 0:
    R = rotx(args.xxx)
    
elif args.yyy != 0 and args.zzz == 0 and args.xxx == 0:
    R = roty(args.yyy)
    
elif args.zzz != 0 and args.xxx == 0 and args.yyy == 0:
    R = rotz(args.zzz)

elif args.xxx != 0 and args.yyy != 0 and args.zzz == 0 :
    R = np.dot(roty(args.yyy),rotx(args.xxx))

elif args.xxx != 0 and args.zzz != 0 and args.yyy == 0:
    R = np.dot(rotz(args.zzz),rotx(args.xxx))

elif args.zzz != 0 and args.yyy != 0 and args.xxx == 0:
    R = np.dot(roty(args.yyy),rotz(args.zzz))

elif args.xxx != 0 and args.yyy != 0 and args.zzz != 0 :
    print('its in the right place');
    R = np.dot(np.dot(roty(args.yyy),rotx(args.xxx)), rotz(args.zzz))

else:
    R = [[1,0,0],[0,1,0],[0,0,1]]

##########gimbal calibration#######

K=np.array([[2359.79036,0, 2031],[ 0, 2359.30434,   1046.5],[ 0 ,  0,   1]])
#K= np.array([[  111356.98038,  0, 2047.50000],[0 ,    141542.78624,   1079.50000],[0, 0, 1 ]])
H1 = np.dot(np.dot(K, R), np.linalg.inv(K)) #

w = 3816#4035#4096#1327#4035#4059
h = 2138#2093#2160#655#2005 #2004
corners_4 = np.array([[1,1], [w,1],[1,h],[w,h]], dtype=np.float32)

#2nd lest error
#mtx = np.array([[2356.17966, 0, 2003.46011],[0 , 2350.96773, 1034.29550],[0, 0, 1 ]])
#dist = np.array([-0.01093,  -0.00486,   -0.00326,   -0.00214,  0.00000])


#best lesat error
#mtx = np.array([[2356.08163, 0, 2001.76580],[0 , 2351.35322, 1027.03103],[0, 0, 1 ]])
#dist = np.array([ -0.01384,   0.00707,   -0.00416,   -0.00254,  0.00000 ])


#best least error based on 2.7 2020 dataset
#mtx = np.array([[29954.31217, 0,  2002.39642],[0 ,  101303.04520,  -352.20279],[0, 0, 1 ]])
#dist = np.array([ -3.52492,   -26.73592,   0.05440,   0.12978,  0.00000])

#webodm
#mtx = np.array([[ 1842.66879753,  0, 263.798671352],[0 , 3494.24601605,-483.752782545],[0, 0, 1 ]])
#dist = np.array([ -0.045682671180243724,   -0.004039330808015797,  0.00809223528363741,   -0.00611818388794988,  0.004976309604304051 ])


#6.8 calibration first
#mtx = np.array([[ 105795.68952,  0, 2047.50000],[0 ,  134018.78419 , 1079.50000],[0, 0, 1 ]])
#dist = np.array([ -33.92024,   -19774.98725,   -0.00682,   0.02145,  0.00000 ])

#6.8 calibration lesat error
#mtx = np.array([[  111356.98038,  0, 2047.50000],[0 ,    141542.78624,   1079.50000],[0, 0, 1 ]])
#dist = np.array([-25.70504,   -67218.08872,   -0.01191,   0.02291,  0.00000])



#11.9 calibration lesat error
#mtx = np.array([[  2346.37769,  0, 2013.51246],[0 ,    2345.73380,   1059.99671],[0, 0, 1 ]])
#dist = np.array([-0.01240, -0.01032, -0.00047, -0.00011, 0.00000])


# 25.5 calibration least error GRACE's 16.5 calibration videos
#mtx = np.array([[  3144.66905,  0, 1916.58117 ],[0 , 3144.67696,  1018.33331],[0, 0, 1 ]])
#dist = np.array([[ 0.02034, -0.02333, -0.00542, -0.00069, 0.00000 ]])



# 23.20.10 calibration grace automate python 
mtx = np.array([[ 7003.06025,  0.00000000,  1726.14480], [ 0.00000000,  7049.87304, -97.2162886], [ 0.00000000,  0.00000000,  1.00000000]])
dist = np.array([[ 0.06693996, -0.15926691, -0.01767889, -0.00425557,  0.34429158]])
 

'''
this is theia camera calibration that was done 19.11.2024

the output from the script:
    ('Camera matrix:\n', array([[2.69065481e+04, 0.00000000e+00, 2.74787941e+03],
       [0.00000000e+00, 2.68572137e+04, 1.57485262e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))
    ('Distortion coefficients:\n', array([[-2.54857725e+00, -1.90431520e+02, -1.70987623e-03,
        -1.14733905e-02,  2.66256271e+04]]))  


'''
'''
#theia calibration
mtx = np.array([[2.69065481e+04, 0.00000000e+00, 2.74787941e+03],
       [0.00000000e+00, 2.68572137e+04, 1.57485262e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-2.54857725e+00, -1.90431520e+02, -1.70987623e-03,
        -1.14733905e-02,  2.66256271e+04]])
'''
#upside data
#mtx = np.array([[4.91398623e+04, 0.00000000e+00, 1.82741811e+03],[0.00000000e+00, 4.88346561e+04, 1.18160716e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#dist = np.array([[-2.28909385e+00,  2.86751816e+03,  1.06471551e-02, 1.87582923e-02,  3.94093036e+00]])


corr = cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), (H1))
#print('corr: ', corr)

#if np.min(corr[:,:,0])<0 and np.min(corr[:,:,1])<0:
offset_matrix = np.array([[1,0, -np.min(corr[:,:,0])],[0,1, -np.min(corr[:,:,1])],[0,0,1]])
width = np.max(corr[:,:,0])-np.min(corr[:,:,0])
height = np.max(corr[:,:,1])-np.min(corr[:,:,1])
corr[:,:,0] = corr[:,:,0]-np.min(corr[:,:,0])
corr[:,:,1] = corr[:,:,1]-np.min(corr[:,:,1])

if np.min(corr[:,:,0])<0:
    offset_matrix = np.array([[1,0, -np.min(corr[:,:,0])],[0,1, 0],[0,0,1]])
    width = np.max(corr[:,:,0])-np.min(corr[:,:,0])
    height = np.max(corr[:,:,1])-np.min(corr[:,:,1])

    corr[:,:,0] = corr[:,:,0]-np.min(corr[:,:,0])
    corr[:,:,1] = corr[:,:,1]-np.min(corr[:,:,1])

if np.min(corr[:,:,0])<0 and np.min(corr[:,:,1])<0:
    offset_matrix = np.array([[1,0, 0],[0,1, -np.min(corr[:,:,1])],[0,0,1]])
    width = np.max(corr[:,:,0])-np.min(corr[:,:,0])
    height = np.max(corr[:,:,1])-np.min(corr[:,:,1])

    corr[:,:,0] = corr[:,:,0]-np.min(corr[:,:,0])
    corr[:,:,1] = corr[:,:,1]-np.min(corr[:,:,1])


#width = np.max(corr[:,:,0])-np.min(corr[:,:,0])
#height = np.max(corr[:,:,1])-np.min(corr[:,:,1])

#corr[:,:,0] = corr[:,:,0]-np.min(corr[:,:,0])
#corr[:,:,1] = corr[:,:,1]-np.min(corr[:,:,1])
#print(corr[:,:,0])
w = np.sort(corr[:,:,0], axis=0)
h = np.sort(corr[:,:,1], axis=0)

w1 = int(np.ceil(w[1]))
w2 = int(np.floor(w[2]))
h1 = int(np.ceil(h[1]))
h2 = int(np.floor(h[2]))

#print(w1,w2,h1,h2)

len_fname = len(args.image_path)

for fname in images:
    print(counter, fname)
    # fnematext = fname[];
    img = cv2.imread(fname)

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    #print(offset_matrix)
    #print(H1)
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    
    #print(width, height)
    print(dst.shape)
    #dst2 = cv2.warpPerspective(dst, np.dot(offset_matrix,H1), (width, height))
 
    #dst3 = dst2[h1:h2 , w1:w2]
    
    #print(calibrated_paths+fname[len_fname:]) 
    cv2.imwrite(calibrated_paths+fname[len_fname:],dst)
    counter += 1
 
cv2.destroyAllWindows()












