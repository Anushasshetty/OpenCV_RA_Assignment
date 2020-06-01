
#Perform the following if you haven't already
# pip install opencv
# pip install matplotlib
# pip install numpy 
# (might also ask you to install nose and tornado)

#Library Imports

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Template initialization, characterization and method definition
template = cv.imread('images/template.jpg',0)
w, h = template.shape[::-1]
method = eval('cv.TM_CCORR_NORMED')

#Running through all the images in the folder (max limit set at 50)
for img_name_flag in range(1,50):
    img_name = 'images/'+ str(img_name_flag)+'.png'
    img = cv.imread(img_name,0)

    try: 
        img2 = img.copy()
    except:
        print('End of images!')
        break
                
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    #Figuring out coordinates and details of the rectangle showing matched area
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    p = cv.rectangle(img,top_left, bottom_right, 0, 5)
    percent_match = (min_val*100)

    #Plotting result matrix and detected points side by side
    cv.putText(p, str(round(percent_match,1))+'%', (top_left[0], top_left[1]-10), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Result Matrix'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detection'), plt.xticks([]), plt.yticks([])
    plt.suptitle(img_name)

    #Save the resulting plot as png in 'results' folder
    plt.savefig('results/result_'+str(img_name_flag)+'.png')
    
    #Uncomment the following only if uou want to see pop up plots
    #plt.show()
