from deepface import DeepFace
import cv2 as cv
from sympy import false
import numpy as np
import asyncio
from app.db_connections.qdrant import qdrant_client  # Import the Qdrant client

video_file = "/home/playground/candidate_verification/videos/Chandrababu Naidu Interview ｜ Green Hydrogen, Drone City： At Davos, C Naidu's To-Do List For Andhra [pV5gmmFL318].webm"
image_file = "\\wsl.localhost\Ubuntu-24.04\home\playground\candidate_verification\images\ChandraBabuNaidu.jpg"  # Path to the image to be verified

cap = cv.VideoCapture(video_file)

# Extract face embedding from the image
image_face = DeepFace.extract_faces(image_file, detector_backend="mtcnn", enforce_detection=false)
if image_face:
    image_embedding = DeepFace.represent(image_face[0]['face'], model_name='Facenet512', enforce_detection=False)[0]['embedding']
else:
    raise ValueError("No face detected in the image.")

async def recognize_faces(frame, faces, image_embedding):
    for face in faces:
        face_embedding = DeepFace.represent(face['face'], model_name='Facenet512', enforce_detection=False)
        query_vector = face_embedding[0]['embedding']
        result = await qdrant_client.query_collection('your_collection_name', query_vector)
        
        if result:
            name = result[0].payload.get('name', 'Unknown')
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Draw rectangle around the face
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put the name text above the rectangle
            cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Compare with the image embedding
            if np.linalg.norm(np.array(query_vector) - np.array(image_embedding)) < 0.6:  # Adjust threshold as needed
                cv.putText(frame, "Match Found", (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=false)
    
    if faces:
        asyncio.run(recognize_faces(frame, faces, image_embedding))
    
    # Display the frame with the face area and name
    cv.imshow('Video', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()