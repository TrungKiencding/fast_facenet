from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import tensorflow as tf
from sqlalchemy.orm import Session
from mtcnn import MTCNN
from database import get_db
from models import Face
from PIL import Image
import numpy as np
import io
import cv2
from face_services import FaceRegistrationService, FaceRecognitionService

app = FastAPI()

# Initialize MTCNN detector
detector = MTCNN()

# Load FaceNet TFLite model
interpreter = tf.lite.Interpreter(model_path="fast_facenet.tflite")
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Input image size for the model
required_size = (input_shape[1], input_shape[2])


def preprocess_face(face_img):
    """Preprocess face for FaceNet model"""
    face_img = cv2.resize(face_img, required_size)
    face_img = face_img.astype('float32')
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def get_embedding(face_pixels):
    """Extract embedding from face image using TFLite model"""
    face_pixels = face_pixels.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], face_pixels)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])
    return embedding[0]

def compare_faces(embedding, threshold, faces):
    """Compare embedding with database"""
    norm_embedding = embedding / np.linalg.norm(embedding)
    best_similarity = -1.0  
    identity = "unknown"
    for face in faces:
        db_embedding_front = np.frombuffer(face.embedding_front, dtype=np.float32)
        db_embedding_left = np.frombuffer(face.embedding_left, dtype=np.float32)
        db_embedding_right = np.frombuffer(face.embedding_right, dtype=np.float32)
        norm_db_embedding_front = db_embedding_front / np.linalg.norm(db_embedding_front)
        norm_db_embedding_left = db_embedding_left / np.linalg.norm(db_embedding_left)
        norm_db_embedding_right = db_embedding_right / np.linalg.norm(db_embedding_right)
        similarity_front = np.dot(norm_embedding, norm_db_embedding_front)
        similarity_left = np.dot(norm_embedding, norm_db_embedding_left)
        similarity_right = np.dot(norm_embedding, norm_db_embedding_right)
        if similarity_front > best_similarity:
            best_similarity = similarity_front
            identity = face.name
        if similarity_left > best_similarity:
            best_similarity = similarity_left
            identity = face.name
        if similarity_right > best_similarity:
            best_similarity = similarity_right
            identity = face.name
    if best_similarity < threshold:
        identity = "unknown"
    return identity, float(best_similarity)

@app.get("/")
def root():
    return{"Hello": "This is API for face recognition"}

@app.post("/recognize")
async def recognize_face(image: UploadFile = File(...), db: Session = Depends(get_db)):
    face_recognition_service = FaceRecognitionService(detector=detector, preprocessor=preprocess_face, embedding_generator=get_embedding, comparator=compare_faces)
    try:
        result = await face_recognition_service.recognize(image, db)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during face recognition: {str(e)}"
        )
    return JSONResponse(content=result)

    
@app.post("/register")
async def register_face(image_front: UploadFile = File(...), image_left: UploadFile = File(...), image_right: UploadFile = File(...), name: str = Form(...), id: str = Form(...), db: Session = Depends(get_db)):
    face_registration_service = FaceRegistrationService(detector=detector, preprocessor=preprocess_face, embedding_generator=get_embedding
    )
    result = await face_registration_service.register(
        image_front, image_left, image_right, name, id, db
    )
    return JSONResponse(content=result)


@app.post("/check")
async def check_database(db: Session = Depends(get_db)):
    try:
        # Query all faces from the database
        faces = db.query(Face).all()
        
        # Create a list of face data
        face_data = []
        for face in faces:
            face_data.append({
                "id": face.id,
                "name": face.name
            })
        
        return JSONResponse(content={
            "status": "success",
            "total_faces": len(face_data),
            "faces": face_data
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Database error: {str(e)}"
            }
        )


@app.post("/delete")
async def delete_face(id: str = Form(...), db: Session = Depends(get_db)):
    try:
        # Find the face in database
        face = db.query(Face).filter_by(id=id).first()
        
        if not face:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Face with ID {id} not found"
                }
            )   
        # Delete the face
        db.delete(face)
        db.commit()
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Face with ID {id} deleted successfully"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Database error: {str(e)}"
            }
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

