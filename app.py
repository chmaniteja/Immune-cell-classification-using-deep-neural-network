# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Immune Cell Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Immune Cell Detection")

# Sidebar
st.sidebar.header("Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Cell Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Cell Detection':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                names = res[0].names
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                names = res[0].names
                class_detections_values = []
                for k, v in names.items():
                    class_detections_values.append(res[0].boxes.cls.tolist().count(k))
                # create dictionary of objects detected per class
                classes_detected = dict(zip(names.values(), class_detections_values))
                # st.text(classes_detected)
                if 'basophil' in classes_detected:
                    disease_detected = classes_detected['basophil']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"##### Cell Detected: \n ##### :blue[basophil] : {disease_detected}")
                            st.warning(f"##### Category: \n ##### :blue[Small Eaters]")
                if 'eosinophil' in classes_detected:
                    disease_detected = classes_detected['eosinophil']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"#### Cell Detected: \n ##### :blue[eosinophil] : {disease_detected}")
                            st.warning(f"#### Category :blue[Big Eaters]")
                if 'erythroblast' in classes_detected:
                    disease_detected = classes_detected['erythroblast']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"#### Cell Detected: \n ##### :blue[erythroblast] : {disease_detected}")
                            st.warning(f"#### Category :blue[Small Eaters]")
                if 'ig' in classes_detected:
                    disease_detected = classes_detected['ig']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"#### Cell Detected: \n ##### :blue[basophil] : {disease_detected}")
                            st.warning(f"#### Category :blue[Big Eaters]")
                if 'lymphocyte' in classes_detected:
                    disease_detected = classes_detected['lymphocyte']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"#### Cell Detected: \n ##### :blue[lymphocyte] : {disease_detected}")
                            st.warning(f"#### Category :blue[Small Eaters]")
                if 'monocyte' in classes_detected:
                    disease_detected = classes_detected['monocyte']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"#### Cell Detected: \n ##### :blue[monocyte] : {disease_detected}")
                            st.warning(f"#### Category :blue[Big Eaters]")
                if 'neutrophil' in classes_detected:
                    disease_detected = classes_detected['neutrophil']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"#### Cell Detected: \n ##### :blue[neutrophil] : {disease_detected}")
                            st.warning(f"#### Category :blue[Big Eaters]")
                if 'platelet' in classes_detected:
                    disease_detected = classes_detected['platelet']
                    if disease_detected:
                        if not disease_detected == 0:
                            st.info(f"#### Cell Detected: \n ##### :blue[platelet] : {disease_detected}")
                            st.warning(f"#### Category :blue[Small Eaters]")
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
