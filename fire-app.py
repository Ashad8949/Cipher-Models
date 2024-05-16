from ultralytics import YOLO
import numpy as np
from base import *
from moviepy.editor import *

# Streamlit app layout
st.title('Fire Detection Model')
st.sidebar.title('Settings')

confidence_threshold = st.sidebar.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.3, step=0.05)

media_type = st.sidebar.radio('Select Input Type', ('Webcam', 'Upload Image', 'Upload Video', 'Enter Video URL'))

# Load YOLO model
model = YOLO('weights/fire-pro.pt')


if media_type == 'Upload Image':
    image_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        image = np.array(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, -1)
        image_with_boxes = image.copy()
        predictions = detect_objects(model, image, confidence_threshold)
        image_with_boxes = draw_boxes(image_with_boxes, predictions)
        st.image(image_with_boxes, channels='BGR')

elif media_type == 'Upload Video':
    video_file = st.sidebar.file_uploader('Upload Video', type=['mp4'])
    if video_file is not None:
        video_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_temp_path.write(video_file.read())
        display_video_with_objects(video_temp_path.name, model, confidence_threshold)
        # Close the video file before deleting
        video_temp_path.close()
        os.unlink(video_temp_path.name)

elif media_type == 'Enter Video URL':
    video_url = st.sidebar.text_input('Enter Youtube Video URL')
    if st.sidebar.button('Submit'):
        play_video(video_url, model)

elif media_type == 'Webcam':
    st.sidebar.write('Waiting for webcam to start...')
    camera_source = st.sidebar.selectbox('Select Camera Source', options=['Laptop Camera', 'DroidCam'])

    if camera_source == 'Laptop Camera':
        camera_index = 0
    elif camera_source == 'DroidCam':
        # Assuming DroidCam uses a different camera index, typically 1 or a custom IP camera address
        camera_index = st.sidebar.text_input('Enter DroidCam Camera Index or IP Address', value='1')

    if st.sidebar.button('Start Webcam'):
        detect_realtime(model, camera_index)

# http://192.168.1.6:4747/video