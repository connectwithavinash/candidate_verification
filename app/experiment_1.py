import cv2 as cv
from deepface import DeepFace
from qdrant import qdrant_client
import asyncio
import concurrent.futures 


video_file = "D:/1_Playground/face_recognition/experiments/videos/Chandrababu Naidu Interview ｜ Green Hydrogen, Drone City： At Davos, C Naidu's To-Do List For Andhra [pV5gmmFL318].webm"

cap = cv.VideoCapture(video_file)


async def process_frame(frame):
    
    try:####time consuming process
        results = await asyncio.to_thread(DeepFace.represent, frame, detector_backend="yolov8", model_name="Facenet", enforce_detection=False)
        ####

        if results:
            if isinstance(results, list):
                for r in results:
                    face_embeddings = r.get('embedding')
                    
                    result = await qdrant_client.query_collection('facial-recognition', face_embeddings)

                    if result:
                        if result[0].score > 0.55:
                            name = result[0].payload.get('name', 'Unknown')
                            facial_area = r.get('facial_area')

                            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 1)

                            cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 200, 0), 1)  # Added text

                            return frame
                        
                        else:
                            facial_area = r.get('facial_area')

                            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                            
                            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 1)

                            return frame

    except Exception as e:
        print(e)

async def main():
    
   try:
        frame_count = 0
        while True:
            ret, frame = cap.read()  

            if not ret:
                break
            
            frame_count +=1
            if frame_count % 5 != 0:
                continue

            frame = await process_frame(frame)

            cv.imshow("Video Frame", frame) 

            if cv.waitKey(1) & 0xFF == ord('q'): 
                break
            
        cap.release()
        cv.destroyAllWindows() 
   except Exception as e:
       print(e)

if __name__ == "__main__":
    asyncio.run(main())