# taken from
# https://github.com/delphinus1024/vector_scope/blob/master/vector_scope.py
# dataset from https://www.kaggle.com/datasets/shravankumar9892/image-colorization?resource=download

#!/usr/bin/env python

# The MIT License (MIT)
# NTSC vector scope
# Copyright (c) 2016 delphinus1024

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import cv2
import sys
import math
import glob
import os

# scope size
cols = 512
rows = 512

# misc constant values
margin = 10
outerlinewidth = 2
outerlinecolor = (128, 128, 128)
small_tick_ratio = 0.98
large_tick_ratio = 0.95
tick_width = 2
dot_radius = 2
vector_len = 8
al_out_angle = 4.
al_out_rad = 0.05
al_thick = 1

# I/Q angle
rad_iq = 33. * math.pi / 180.

# NTSC constants
a_r =  0.701
b_r = -0.587
c_r = -0.114

a_b = -0.299
b_b = -0.587
c_b =  0.886

maxval = 0

def calc_ryby(r,g,b):
    ry = a_r * r + b_r * g + c_r * b
    by = a_b * r + b_b * g + c_b * b
    return ry,by
    
def calc_ec(ry,by):
    ec = math.sqrt( ((ry/1.14) ** 2) + ((by/2.03) ** 2) )
    return ec

def calc_theta(ry,by):
    theta = math.atan2(ry/1.14, by/2.03)
    return theta

def calc_maxval():
    global maxval
    ry = a_r
    by = c_b
    maxval = calc_ec(ry,by)
    return maxval
    
def calc_transform(r,g,b):
    ry,by = calc_ryby(r,g,b)
    ec = calc_ec(ry,by)
    theta = calc_theta(ry,by)
    return ec,theta

def pole2cart(center_x,center_y,theta,radius):
    x = np.float64(center_x) + np.float64(radius) * math.cos(theta)
    y = np.float64(center_y) - np.float64(radius) * math.sin(theta)
    return int(x),int(y)

def rgb2cart(center_x,center_y,radius,r,g,b,angle_delta):
    global maxval
    ec,theta = calc_transform(r,g,b)
    x,y = pole2cart(center_x,center_y,theta + (angle_delta * math.pi / 180.) ,ec / maxval * radius)
    return x,y

def draw_allowance(result_img,center_x,center_y,radius,v,c,fontType):
    #inner
    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.0474),v[0],v[1],v[2],0.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.0474),v[0],v[1],v[2],0.)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.0474),v[0],v[1],v[2],2.5)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.0474),v[0],v[1],v[2],2.5)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)
    
    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.0474),v[0],v[1],v[2],-2.5)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.0474),v[0],v[1],v[2],-2.5)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)
    
    x ,y  = rgb2cart(center_x,center_y,radius * (1.),v[0],v[1],v[2],-2.5)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1.),v[0],v[1],v[2], 2.5)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.0474),v[0],v[1],v[2],-2.5)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. - 0.0474),v[0],v[1],v[2], 2.5)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. + 0.0474),v[0],v[1],v[2],-2.5)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.0474),v[0],v[1],v[2], 2.5)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    # outer
    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.2),v[0],v[1],v[2],-10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. - 0.2),v[0],v[1],v[2],-10. + al_out_angle)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.2),v[0],v[1],v[2],-10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. - 0.2 + al_out_rad),v[0],v[1],v[2],-10.)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.2),v[0],v[1],v[2],10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. - 0.2),v[0],v[1],v[2],10. - al_out_angle)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. - 0.2),v[0],v[1],v[2],10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. - 0.2 + al_out_rad),v[0],v[1],v[2],10.)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. + 0.2),v[0],v[1],v[2],-10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.2),v[0],v[1],v[2],-10. + al_out_angle)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. + 0.2),v[0],v[1],v[2],-10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.2 - al_out_rad),v[0],v[1],v[2],-10.)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. + 0.2),v[0],v[1],v[2],10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.2),v[0],v[1],v[2],10. - al_out_angle)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius * (1. + 0.2),v[0],v[1],v[2],10.)
    x2,y2 = rgb2cart(center_x,center_y,radius * (1. + 0.2 - al_out_rad),v[0],v[1],v[2],10.)
    cv2.line(result_img,(x,y), (x2,y2),outerlinecolor,al_thick)

    x ,y  = rgb2cart(center_x,center_y,radius,v[0],v[1],v[2],0.)
    cv2.putText(result_img,c, (x + vector_len,y - vector_len), fontType,1, outerlinecolor, 1, 16)

def draw_backgroud(result_img,center_x,center_y,radius):
    # outer circle and tick
    cv2.circle(result_img, (center_x, center_y), radius,outerlinecolor, outerlinewidth, 8)
    
    radius_div = radius / 5.
    for i in range(1,5):
        cv2.circle(result_img, (center_x, center_y),int(float(i) * radius_div),outerlinecolor, 1)
        
    cv2.line(result_img,(center_x - radius, center_y),(center_x + radius, center_y),outerlinecolor,1)
    cv2.line(result_img,(center_x, center_y - radius),(center_x, center_y + radius),outerlinecolor,1)
    
    for i in range(0,360,2):
        theta = np.float64(i) / 180 *  math.pi
        if (i % 10) == 0:
            r_s = np.float64(radius) * large_tick_ratio
        else:
            r_s = np.float64(radius) * small_tick_ratio
            
        xs,ys = pole2cart(center_x,center_y,theta,radius)
        xe,ye = pole2cart(center_x,center_y,theta,r_s)
        cv2.line(result_img, (xs,ys), 
            (xe,ye),outerlinecolor,tick_width)
            
    # I/Q lines
    xs,ys = pole2cart(center_x,center_y,rad_iq,radius)
    xe,ye = pole2cart(center_x,center_y,math.pi + rad_iq,radius)
    cv2.line(result_img, (xs,ys), 
        (xe,ye),outerlinecolor,1)

    xs,ys = pole2cart(center_x,center_y,math.pi * 0.5 + rad_iq,radius)
    xe,ye = pole2cart(center_x,center_y,math.pi * 1.5 + rad_iq,radius)
    cv2.line(result_img, (xs,ys), 
        (xe,ye),outerlinecolor,1)
    
    # draw vectors
    vec = [[0.0,0.0,1.0],[0.0,1.0,0.0],[0.0,1.0,1.0],
        [1.0,0.0,0.0],[1.0,0.0,1.0],[1.0,1.0,0.0]]
    col_name = ["B","G","CY","R","MG","YL"]
    
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    
    for v,c in zip(vec,col_name):
        draw_allowance(result_img,center_x,center_y,radius,v,c,fontType)
        
def draw_pixel(result_img,center_x,center_y,radius,bgr):
    global dot_radius
    x,y = rgb2cart(center_x,center_y,radius,bgr[2],bgr[1],bgr[0],0.)
    col = (int(bgr[0] * 255.),int(bgr[1] * 255.),int(bgr[2] * 255.))
    cv2.circle(result_img, (x,y), dot_radius,col,-1)
    

def plot_vector_scope(image, filename = "result.txt", prefix = "data"):
    # image = cv2.imread(filename)
    height, width, depth = image.shape
    print("width=",width," height=",height," depth=",depth)

    # accept only rgb
    if depth != 3:
        print("Error. Not RGB image.")
        quit()

    # initialize table
    maxval = calc_maxval()

    # create result image in black bg
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    radius = int(rows / 2 - margin)
    result_img = np.zeros((rows, cols, 3), np.uint8) 

    # reshape to 1d & hsv conversion
    lin_image = (np.reshape(image,(height * width,3))).astype(np.float64) / 255.

    # plot all pixels
    for bgr in lin_image:
        draw_pixel(result_img,center_x,center_y,radius,bgr)
        
    # background
    draw_backgroud(result_img,center_x,center_y,radius)

    name1 = f"{prefix}/vectorscopes/{filename}.png"
    cv2.imwrite(name1, result_img)
    print(f"image size = {image.shape}, result_img size = {result_img.shape}")
    print(f"image size = {image.shape[0] < result_img.shape[0]} and {image.shape[1] <= result_img.shape[1]}")
    if image.shape[0] >= result_img.shape[0] and image.shape[1] >= result_img.shape[1]:
        print("first")
        pad_width = (image.shape[0] - result_img.shape[0]) 
        pad_height = (image.shape[1] - result_img.shape[1])
        print(pad_width, pad_height)
        result_img = np.pad(result_img, ((pad_width // 2, pad_width - pad_width // 2), (pad_height // 2, pad_height - pad_height // 2), (0, 0)), 'constant', constant_values=0)
    elif image.shape[0] < result_img.shape[0] and image.shape[1] < result_img.shape[1]:
        print("second")
        pad_width = (result_img.shape[0] - image.shape[0])
        pad_height = (result_img.shape[1] - image.shape[1])
        print(pad_width, pad_height)
        image = np.pad(image, ((pad_width // 2, pad_width - pad_width // 2), (pad_height // 2, pad_height - pad_height // 2), (0, 0)), 'constant', constant_values=0)
    elif image.shape[0] >= result_img.shape[0] and image.shape[1] <= result_img.shape[1]:
        print("third")
        pad_width = (image.shape[0] - result_img.shape[0])
        pad_height = (result_img.shape[1] - image.shape[1])
        print(pad_width, pad_height)
        result_img = np.pad(result_img, ((pad_width // 2, pad_width - pad_width // 2), (0, 0), (0, 0)), 'constant', constant_values=0)
        image = np.pad(image, ((0 , 0), (pad_height // 2, pad_height - pad_height // 2), (0, 0)), 'constant', constant_values=0)
    else:
        print("last")
        pad_width = (result_img.shape[0] - image.shape[0])
        pad_height = (image.shape[1] - result_img.shape[0])
        print(pad_width, pad_height)
        result_img = np.pad(result_img, ((0 , 0), (pad_height // 2, pad_height - pad_height // 2), (0, 0)), 'constant', constant_values=0)
        image = np.pad(image, ((pad_width // 2, pad_width - pad_width // 2), (0, 0), (0, 0)), 'constant', constant_values=0)
        

    numpy_horizontal_concat = np.concatenate((image, result_img), axis=1)
    name2 = f"{prefix}/vectorscopes_and_inputs/{filename}.png"
    cv2.imwrite(name2, numpy_horizontal_concat)
    # cv2.imshow("result", numpy_horizontal_concat)

# import a image file
def main():
    argvs = sys.argv
    argc = len(argvs)

    if argc < 2:
        user_input = "*"
    else:
        user_input = argvs[1]
    
    prefix = "data/imgs_out"
    pattern = f"{prefix}/{user_input}.png"
    print(f"Searched pattern: {pattern}")
    filenames = glob.glob(pattern)

    print(f"Found {len(filenames)} files")
    for filename in filenames:
        image = cv2.imread(filename)
        filename = filename.split("\\")[-1].replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
        print(f"Processing {filename}")
        plot_vector_scope(image, filename, prefix=prefix)

if __name__ == '__main__':
    main()