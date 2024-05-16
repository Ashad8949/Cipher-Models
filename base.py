import os
import cv2
import math
import smtplib
import tempfile
import subprocess
import streamlit as st
from pytube import YouTube
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from moviepy.video.io.VideoFileClip import VideoFileClip


# Function to perform object detection
def detect_objects(model, image, confidence_threshold):
    results = model(image, stream=True)
    predictions = []
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0]
            if confidence > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                prediction = {
                    "x": x1 + width / 2,
                    "y": y1 + height / 2,
                    "width": width,
                    "height": height,
                    "confidence": confidence,
                    "class": model.names[int(box.cls[0])],
                    "class_id": int(box.cls[0])
                }
                predictions.append(prediction)
    return predictions



# Function to draw bounding boxes on the frame
def draw_boxes(frame, predictions):
    for prediction in predictions:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{prediction['class']} {prediction['confidence']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


# Function to display video with detected objects
def display_video_with_objects(video_path, model, confidence_threshold):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    stframe = st.empty()
    json_result = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        predictions = detect_objects(model, frame, confidence_threshold)
        # Draw bounding boxes on the frame
        frame_with_boxes = draw_boxes(frame.copy(), predictions)
        # Display the video frame with bounding boxes
        stframe.image(frame_with_boxes, channels='BGR', use_column_width=True)
        # Display the JSON results
        json_result.write({"predictions": predictions})



def play_video(video_url, model):
    if video_url:
        try:
            yt = YouTube(video_url)
            video_stream = yt.streams.filter(file_extension="mp4").first()
            temp_folder = "temp_video"
            os.makedirs(temp_folder, exist_ok=True)
            video_path = os.path.join(temp_folder, "temp_video.mp4")
            video_stream.download(output_path=temp_folder, filename="temp_video.mp4")
            video_clip = VideoFileClip(video_path)
            print(video_path)
            display_video_with_objects(video_path, model=model, confidence_threshold=0.3)
        except Exception as e:
            st.error(f"Error: {e}")


def detect_realtime(model, video_source):
    # Email Configuration
    email_address = 'ashad8949@gmail.com'
    email_password = 'pyjm lacw gqfw yaka'
    recipient_email = 'alamashad507@gmail.com'

    # Open the video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Failed to connect to video source")
        st.stop()

    stop_button = st.button("Stop")  # Unique key for the button
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from video source")
            break

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        # Getting bbox, confidence and class names informations to work with
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                print(confidence)
                if confidence > 70:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

                    # Sending email with the detected fire picture attached
                    msg = MIMEMultipart()
                    msg['From'] = email_address
                    msg['To'] = recipient_email
                    msg['Subject'] = 'Fire Detected'

                    body = 'Fire detected at the location.'
                    msg.attach(MIMEText(body, 'plain'))

                    # Save the frame with fire detected
                    img_path = os.path.join(os.getcwd(), 'fire_detected.jpg')
                    cv2.imwrite(img_path, frame)

                    # Attach the image to the email
                    with open(img_path, 'rb') as attachment:
                        image = MIMEImage(attachment.read())
                        attachment.close()
                        msg.attach(image)

                    # Send the email
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(email_address, email_password)
                    server.sendmail(email_address, recipient_email, msg.as_string())
                    server.quit()

                    st.write("Email Sent!")  # Show message in the Streamlit app

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Quit the loop if 'Stop' button is clicked
        if stop_button:
            break

    # Release the video capture when finished
    cap.release()