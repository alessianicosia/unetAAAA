# TASK 1
# Pre-processing to realize binary images and to re-find coordinates from binary images


import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
import cv2
import numpy as np
import xmltodict
from matplotlib import pyplot
from scipy import interpolate
from matplotlib import image
from PIL import Image


# Pass DICOM image, print information, plot and save
ds = dcmread(r'C:\Users\aless\Downloads\C5-1_cine_1.dcm')
print(f"Image size.......: {ds.Rows} x {ds.Columns}")
print(f"Samples per pixels.......: {ds.SamplesPerPixel}")
fig = plt.figure(num=1, clear = False)
plt.imshow(ds.pixel_array[0,:,:,0], cmap=plt.get_cmap('gray'))
pyplot.title("DICOM image")
fig = plt.figure(num=1, clear = False)
cv2.imwrite('00_training.jpg', ds.pixel_array[0,:,:,0])


# Open xml file and transform coordinates in matrix (17x2)
with open(r'C:\Users\aless\Downloads\C5-1_cine_1_1.2.840.113663.1500.1.420173227.3.11.20210604.131530.903_quantifs.xml') as fd:
    doc = xmltodict.parse(fd.read())
a = doc['AAA_BioMech_Report']['tracked_pts_str']
split_str = a.split(',')[0:74]
ints = [int(float(s)) for s in split_str]
matrix=np.reshape(ints, (37,2))
print(matrix)
x = matrix[:,0]
y = matrix[:,1]


# Spline interpolation
x = np.r_[x, x[0]]
y = np.r_[y, y[0]]
tck, u = interpolate.splprep([x, y], s=0, per=True)
xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
a = np.array(xi)
b = np.array(yi)
matrix1 = np.reshape(a, (1000,1))
matrix2 = np.reshape(b, (1000,1))
matrix3 = np.hstack((matrix1, matrix2))
box = np.int0(matrix3)


# Realize binary image, plot and save
canvas = np.zeros((ds.Rows,ds.Columns,ds.SamplesPerPixel), np.uint8)
cv2.drawContours(canvas, [box], -1, (255,255,255), -1)
#cv2.polylines(canvas, [matrix], isClosed=True, color=(0,0,0), thickness=2)
cv2.imwrite('00_manual1.png', canvas[:,:,0])
fig = plt.figure(num=2, clear = False)
plt.imshow(canvas)
pyplot.title("Binary image 1")


# Cropping
im = Image.open(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_manual1.png')
im.crop((380, 200, 680, 500)).save(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_manual1.png', quality=95)
im = Image.open(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_training.jpg')
im.crop((380, 200, 680, 500)).save(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_training.jpg', quality=95)


# Comparison DICOM vs. Binary image, plot and save (sum)
img1 = cv2.imread(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_manual1.png')
img2 = cv2.imread(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_training.jpg')
img3 = cv2.hconcat([img1, img2])
fig = plt.figure(num=3, clear = False)
plt.imshow (img3)
pyplot.title("Comparison 1")
cv2.imwrite('Comparison 1.png', img3)


# Coordinates extraction from binary image
img1 = cv2.imread(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_manual1.png')
imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #convert an image from one color space to another
ret, thresh = cv2.threshold(imgray, 128, 255, 0) #thresholding replace each pixel in an image with a black pixel
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#find contours (points) and levels of hierarchy (contour retrieval mode), then appoximation method
print(contours, hierarchy)


# Realize binary image from found cooordinates
canvas2 = np.zeros((img1.shape), np.uint8)
cv2.drawContours(canvas2, contours, -1, (255,255,255), -1)
cv2.polylines(canvas2, contours, isClosed=True, color=(0,0,0), thickness=2)
fig = plt.figure(num=4, clear = False)
plt.imshow (canvas2)
pyplot.title("Binary image 2")
cv2.imwrite('00_manual2.png', canvas2)


# Comparison Binary image 1 vs. Binary image 2, plot and save (sum binary images)
img1 = cv2.imread(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_manual1.png')
img4 = cv2.imread(r'C:\Users\aless\Desktop\Tesi\Python\unet-master\00_manual2.png')
img5 = cv2.hconcat([img1, img4])
fig = plt.figure(num=5, clear = False)
plt.imshow(img5)
pyplot.title("Comparison 2")
cv2.imwrite('Comparison 2.png', img5)


# Visualize sovrapposition (first and second segmentation on DICOM image)
img = cv2.imread('00_training.jpg')
canvas3 = img.copy()
Results = cv2.polylines(canvas3, [matrix], isClosed=True, color=(255,0,0), thickness=2)
Results_2 = cv2.polylines(canvas3, contours, isClosed=True, color=(255,0,0), thickness=2)
img6 = cv2.hconcat([Results, Results_2])
fig = plt.figure(num=6, clear = False)
plt.imshow(img6)
pyplot.title("Comparison of results")
cv2.imwrite('Comparasion of results.png', img6)

plt.show()