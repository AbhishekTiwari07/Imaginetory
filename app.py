import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import copy

st.set_page_config(
    page_title="Image Enhancement",
    page_icon="img/edit.png",
)


@st.cache
def load_image(img):
    im = Image.open(img)
    return im

def upscale(img):
    img = np.array(img)
    bipolar = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    return bipolar

def blurring(img, value):
    value = int(value)*3
    blur_filter = np.ones((value, value), np.float)/(value*value)
    image_blur = cv2.filter2D(img, -1, blur_filter)

    return image_blur

def denoising(img):
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    dst = cv2.fastNlMeansDenoisingColored(rgb_img,None,10,10,7,21)
    b,g,r = cv2.split(dst)
    rgb_dst = cv2.merge([r,g,b])
    return rgb_dst

def negate(img):

    result = copy.deepcopy(img)
    total_channels = result.shape[2]

    bytes = result.itemsize
    MAX_PIXEL_VAL = (2 ** (bytes*8))-1
    for i in range(total_channels):
        result[:, :, i] = MAX_PIXEL_VAL-result[:, :, i]

    return result

def gaussianBlur(img, value):
    img = cv2.medianBlur(img, value)
    return img

def medianBlur(img, value):
    return cv2.GaussianBlur(img, (value,value), cv2.BORDER_DEFAULT)

def app():
    activities = ['Enhancements', 'Face Detection']
    choice = st.sidebar.selectbox('Select activities', activities)

    if choice == 'Enhancements':
        st.title('Dashboard')
        image_file = st.file_uploader(
            "Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is None:
            st.info('Uplaod image here')
        else:
            if image_file is not None:
                col1, col2 = st.columns(2)
                our_image = Image.open(image_file)
                col1.header('Original Image')
                col1.image(our_image, use_column_width=True)

            enhance_type = st.sidebar.radio('Enhancement Types', [
                'Original', 'Thresholding','Adaptive Thresholding', 'Denoising','Bluring: Median',
                'Bluring: Gaussian', 'Negative' , 'Upscale'])

            if enhance_type == 'Thresholding':
                img = np.array(our_image)
                threshold = st.sidebar.slider('Threshold', 0, 255, 1)
                out_img = gaussianBlur(img, threshold)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)

            if enhance_type == 'Adaptive Thresholding':
                img = np.array(our_image)
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41,3)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)

            if enhance_type == 'Bluring: Median':
                img = np.array(our_image)
                br_rate = st.sidebar.slider('Bluring', 1, 9, 2)
                out_img = medianBlur(img, br_rate)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)
            
            if enhance_type == 'Bluring: Gaussian':
                our_new_image = np.array(our_image)
                br_rate = st.sidebar.slider('Bluring', 1, 9, 2)
                image = gaussianBlur(our_new_image, br_rate)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)

            elif enhance_type == 'Negative':
                our_image = np.array(our_image)
                out_img = negate(our_image)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)

            elif enhance_type == 'Upscale':
                our_image = np.array(our_image)
                image = upscale(our_image)
                col2.header('Edited Image')
                col2.image(image, use_column_width=True)

            elif enhance_type == 'Denoising':
                our_image = np.array(our_image)
                image = denoising(our_image)
                col2.header('Edited Image')
                col2.image(image, use_column_width=True)
    elif choice == 'Face Detection':
        pass

if __name__ == "__main__":
    app()
