import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline #Only for notebook
import cv2
import torch
from models import Net
from torch.autograd import Variable
net = Net()
net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))
net.eval()

# load in color image for face detection
image = cv2.imread('images/obamas.jpg')
# switch red and blue color channels
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plot the image
fig = plt.figure(figsize=(9,9))
plt.imshow(image)
# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)
# make a copy of the original image to plot detections on
image_with_detections = image.copy()
# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
fig = plt.figure(figsize=(9,9))
plt.imshow(image_with_detections)
image_copy = np.copy(image)

#loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    #Select the region of interest that is the face in the image
    roi = image_copy[y:y+h, x:x+w]
    #Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    #Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = roi/255.0
    #Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi = cv2.resize(roi, (224, 224))
    image_plot = np.copy(roi)
    roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
    #Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    print (roi.shape)
    roi = np.transpose(roi, (2, 0, 1))
    roi = torch.from_numpy(roi)
    roi = Variable(roi)
    roi = roi.type(torch.FloatTensor)
    roi = roi.unsqueeze(0)
    print (roi.size())
    #Make facial keypoint predictions using your loaded, trained network
    predicted_key_pts = net(roi)
    print (predicted_key_pts.size())
    predicted_key_pts = predicted_key_pts.view( 68, -1)
    predicted_key_pts = predicted_key_pts.data
    predicted_key_pts = predicted_key_pts.cpu().numpy()
    predicted_key_pts = predicted_key_pts*50.0+100
    print (predicted_key_pts.shape)
    #Display each detected face and the corresponding keypoints
    print (image_plot.shape)
    plt.imshow(image_plot, cmap='gray');plt.show()
    plt.imshow(image_plot, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m'); plt.show()
