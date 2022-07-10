from PIL import UnidentifiedImageError
import streamlit as st
import base64
from io import BytesIO
import numpy as np
import cv2

from essentials import Stage1, Stage2
from decay_detect import Stage3



# initializing the objects
stage1 = Stage1()
stage2 = Stage2()
stage3 = Stage3()



# page title config
st.set_page_config(page_title="FRU-IT", page_icon="./img/favicon.png")

#opening custom css
# with open('./Styles/styles.css') as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# title
with st.container():
    # st.write("""
    #     # FRU-IT 🥭
    #     ### Your personal fruit companion
    # """)
    st.markdown("<h1 style='text-align: center; color: white;'>FRU-IT 🥭</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: gray;'>Your personal fruit companion</h2>", unsafe_allow_html=True)

def display_img(img_path):
    st.image(img_path, use_column_width=True)


def display_output(output):
    st.image(output, width=300, use_column_width=True)


def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href



with st.container():
    # file open dialog and function
    
    st.markdown("""---""")
    img_path = st.file_uploader("Upload the image of the mango 🥭", type=["jpg", "png", "jpeg"])
    if img_path is not None:
        st.success("Image uploaded successfully")
        display_img(img_path)
    st.markdown("""---""")
    if st.checkbox('Use camera input'):    
        st.markdown("<h5 style='text-align: center; color: gray;'>Capture your image 📸</h5>", unsafe_allow_html=True)
        img_path = st.camera_input(label="")
            # button to run ripeness_detection()
        st.markdown("""---""")
    
        if img_path is not None:
            st.success("Image uploaded successfully")
            display_img(img_path)

    st.markdown("""---""")
    if img_path is not None:
        if st.button("Detect Ripeness", help="Click to detect the ripeness of the mango"):
            st.markdown("""---""")
            with st.spinner(text="In progress..."):
                prediction = stage1.detect_fruit(img_path)
                _, op_msg = stage1.get_prediction(prediction)
                img_path.seek(0)
                img_array = np.asarray(bytearray(img_path.read()), dtype=np.uint8)

                st.markdown(f"<h4 style='text-align: center; color: whitesmoke;'>{op_msg}</h4>", unsafe_allow_html=True)
                st.markdown("""---""")
                
                if prediction.argmax() == 0:
                    banana_ripeness, op_msg = stage2.detect_banana_ripeness(img_path)
                    st.markdown(f"<h4 style='text-align: center; color: whitesmoke;'>{op_msg}</h4>", unsafe_allow_html=True)
                    st.markdown("""---""")
                    if banana_ripeness == 0 or banana_ripeness == 1:
                        op_img , output = stage3.detect_banana_decay(img_array, None)
                        display_output(op_img)
                elif prediction.argmax() == 1:
                    mango_ripeness, op_msg = stage2.detect_mango_ripeness(img_path)
                    st.markdown(f"<h4 style='text-align: center; color: whitesmoke;'>{op_msg}</h4>", unsafe_allow_html=True)
                    st.markdown("""---""")
                    if mango_ripeness == 0 or mango_ripeness == 1:
                        op_img , output = stage3.detect_mango_decay(img_array, mango_ripeness)
                        display_output(op_img)

                elif prediction.argmax() == 2:
                    papaya_ripeness, op_msg = stage2.detect_papaya_ripeness(img_path)
                    st.markdown(f"<h4 style='text-align: center; color: whitesmoke;'>{op_msg}</h4>", unsafe_allow_html=True)
                    st.markdown("""---""")
                    if papaya_ripeness == 0 or papaya_ripeness == 1:
                        op_img , output = stage3.detect_papaya_decay(img_array, papaya_ripeness)
                        display_output(op_img)
                        

                else:
                    print('💡I Dont know wth is that')
                    
            st.info("Ripeness detection completed")
            st.markdown("""---""")
        