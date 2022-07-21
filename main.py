import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import time
import subprocess

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    filename = secure_filename(file.filename)
    lst = [filename]
    name , extension = os.path.splitext(lst[0])
    p = os.path.join('data/video/afterDetection', filename)
    # print("exteeeeeeeeeeeeeeention",extension)
    # global id , part
    id = request.form['id']
    part = request.form['part']
    # k = list()
    # k.append(id)
    # k.append(part)
    t = os.path.join(app.config['UPLOAD_FOLDER'], 'partid.txt')
    with open(t,'w') as f:
        f.writelines('{}\n{}\n'.format(id,part))

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    elif extension in ['.png','.jpg','.jpeg']:

          subprocess.run(
              ['python', 'detect.py', '--weights', './checkpoints/yolov4-416', '--size', '416', '--model', 'yolov4',
               '--images', os.path.join(app.config['UPLOAD_FOLDER'], filename), '--output', p, '--crop', '--count'])


    else:
        filename = secure_filename(file.filename)
        # nwli l file name nzidha l id wel part w bead ne5ou menha heka lkol
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        print("hneeeeeeeeeeeeee",filename)

        subprocess.run(['python', 'detect_video.py', '--weights', './checkpoints/yolov4-416', '--size', '416', '--model', 'yolov4','--video', os.path.join(app.config['UPLOAD_FOLDER'], filename), '--output', p, '--crop', '--count'])
        # subprocess.run(
        #     ['python', 'detect.py', '--weights', './checkpoints/yolov4-416', '--size', '416', '--model', 'yolov4',
        #      '--images', os.path.join(app.config['UPLOAD_FOLDER'], filename), '--output', p, '--crop', '--count'])
        print("awed affichiih belehi 5ali nraa :",filename)
        return render_template('upload.html', filename=filename)


@app.route('/display/<filename>')
def display_video(filename):
    # print('display_video filename: ' + filename)
    # hnee nml detection : nkolou hez l video men ghadi w hanou andek l path c deja #
    print("choud be heeeeeeeeree",filename)
    return redirect(url_for('data/video/afterDetection/', filename=filename), code=301)

    # return redirect(url_for('./data/video/afterDetection', filename=filename), code=301)


if __name__ == "__main__":
    app.run()