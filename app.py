from flask import Flask, redirect, render_template, request, session, url_for, flash
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os
import shutil

import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

app = Flask(__name__)
dropzone = Dropzone(app)
# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'
plt.switch_backend('Agg')

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB
    
@app.route('/', methods=['GET', 'POST'])
def index():
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
        folder = os.getcwd() + '/uploads'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    # handle image upload from Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )
            # append image urls
            file_urls.append(photos.url(filename))
            
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request    
    return render_template('index.html')

@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)

    #Template initialization, characterization and method definition
    template = cv.imread('uploads/template.jpg',0)
    w, h = template.shape[::-1]
    method = eval('cv.TM_CCORR_NORMED')
    results_plt = []
    for img_name_flag in range(1,50):
      img_name = 'uploads/'+ str(img_name_flag)+'.png'
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
      plt.savefig(os.getcwd() + '/static/results/result_'+str(img_name_flag)+'.png')

      results_plt.append('/static/results/result_'+str(img_name_flag)+'.png')

    return render_template('results.html', file_urls=results_plt)