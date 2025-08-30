import streamlit as st
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # nano model - fastest

st.title("🚀 Real-Time Object Detection")
st.markdown("Detects object at real time and give the Confidence score.")

# Camera start button
run = st.checkbox('🎥 Start Camera')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("❌ Camera not found")
        break

    # Run YOLO detection
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Convert BGR → RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
