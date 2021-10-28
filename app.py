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


def contrast_stretch(img):

    if img.itemsize > 1:
        result = np.zeros(img.shape, dtype=np.uint)

    else:
        result = np.zeros(img.shape, dtype=np.uint8)

    bytes = img.itemsize

    s1 = 0
    s2 = (2 ** (bytes*8))-1

    for k in range(3):
        r1 = img[:, :, k].min()
        r2 = img[:, :, k].max()

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixVal = img[i, j, k]
                newVal = (((s2-s1)*(pixVal-r1))//(r2-r1))+s1

                if newVal > 0:
                    result[i, j, k] = newVal

                else:
                    result[i, j, k] = 0
    return result


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
                'Original', 'Gray scale', 'Contrast', 'Brightness',
                'Bluring', 'Negative', 'Contrast Stretching', 'Upscale'])

          

            if enhance_type == 'Contrast':

                c_rate = st.sidebar.slider('Constrast', -4.5, 4.5, 1.0)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                col2.header('Edited Image')
                col2.image(img_output, use_column_width=True)

            

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

            elif enhance_type == 'Contrast Stretching':

                our_image = np.array(our_image)
                out_img = contrast_stretch(our_image)
                col2.header('Edited Image')
                col2.image(out_img, use_column_width=True)

            elif enhance_type == 'Upscale':
                image = upscale(our_image)
                col2.header('Edited Image')
                col2.image(image, use_column_width=True)


if __name__ == "__main__":
    app()
