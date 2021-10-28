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
\




def negate(img):

    result = copy.deepcopy(img)
    total_channels = result.shape[2]

    bytes = result.itemsize
    MAX_PIXEL_VAL = (2 ** (bytes*8))-1
    for i in range(total_channels):
        result[:, :, i] = MAX_PIXEL_VAL-result[:, :, i]

    return result


def app():
    activities = ['Enhancements']
    choice = st.sidebar.selectbox('Select activities', activities)

    if choice == 'Enhancements':
        st.title('Image Editor')
        image_file = st.file_uploader(
            "Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is None:
            st.info('please upload image!!')
        else:
            if image_file is not None:
                col1, col2 = st.columns(2)
                our_image = Image.open(image_file)
                col1.header('Original Image')
                col1.image(our_image, use_column_width=True)

            enhance_type = st.sidebar.radio('Enhancement Types', [
                'Original', 
                'Bluring', 'Negative' , 'Upscale'])

          


            

            elif enhance_type == 'Bluring':

                our_new_image = np.array(our_image)
                br_rate = st.sidebar.slider('Bluring', 1, 10, 1)
                out_img = blurring(our_new_image, br_rate)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)

            elif enhance_type == 'Negative':

                our_image = np.array(our_image)
                out_img = negate(our_image)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)


            elif enhance_type == 'Upscale':
                image = upscale(our_image)
                col2.header('Edited Image')
                col2.image(image, use_column_width=True)


if __name__ == "__main__":
    app()
