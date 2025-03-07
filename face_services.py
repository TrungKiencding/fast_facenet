from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from PIL import Image
import numpy as np
import io
import cv2
from models import Face
from fastapi.responses import JSONResponse

class FaceRegistrationService:
    def __init__(self, detector, preprocessor, embedding_generator):
        self.detector = detector
        self.preprocessor = preprocessor
        self.embedding_generator = embedding_generator

    def _load_image(self, image_file: UploadFile) -> np.ndarray:
        content = image_file.file.read()
        image = Image.open(io.BytesIO(content))
        return np.array(image)

    def _convert_to_rgb(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3 and img.shape[2] == 3:
            return img
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    def _detect_face(self, image: np.ndarray, position: str):
        faces = self.detector.detect_faces(image)
        if not faces:
            raise HTTPException(status_code=400, detail=f"Face {position} not detected")
        return faces[0]

    def _extract_face(self, image: np.ndarray, face_box: dict) -> np.ndarray:
        x, y, width, height = face_box['box']
        x, y = max(0, x), max(0, y)
        return image[y:y+height, x:x+width]

    async def register(self, image_front: UploadFile, image_left: UploadFile, 
                      image_right: UploadFile, name: str, id: str, db: Session):
        # Process front image
        img_front = self._convert_to_rgb(self._load_image(image_front))
        img_left = self._convert_to_rgb(self._load_image(image_left))
        img_right = self._convert_to_rgb(self._load_image(image_right))

        # Detect faces
        face_front = self._detect_face(img_front, "front")
        face_left = self._detect_face(img_left, "left")
        face_right = self._detect_face(img_right, "right")

        # Extract face regions
        face_front_img = self._extract_face(img_front, face_front)
        face_left_img = self._extract_face(img_left, face_left)
        face_right_img = self._extract_face(img_right, face_right)

        # Generate embeddings
        processed_front = self.preprocessor(face_front_img)
        processed_left = self.preprocessor(face_left_img)
        processed_right = self.preprocessor(face_right_img)

        embedding_front = self.embedding_generator(processed_front)
        embedding_left = self.embedding_generator(processed_left)
        embedding_right = self.embedding_generator(processed_right)

        # Save to database
        existing_face = db.query(Face).filter_by(id=id).first()
        if existing_face:
            self._update_existing_face(existing_face, embedding_front, 
                                     embedding_left, embedding_right, db)
        else:
            self._create_new_face(id, name, embedding_front, embedding_left, 
                                embedding_right, db)

        return {"status": "success", "message": f"Face registered for {name}"}

    def _update_existing_face(self, face, emb_front, emb_left, emb_right, db):
        face.embedding_front = emb_front.tobytes()
        face.embedding_left = emb_left.tobytes()
        face.embedding_right = emb_right.tobytes()
        db.commit()
        db.refresh(face)

    def _create_new_face(self, id, name, emb_front, emb_left, emb_right, db):
        new_face = Face(
            id=id,
            name=name,
            embedding_front=emb_front.tobytes(),
            embedding_left=emb_left.tobytes(),
            embedding_right=emb_right.tobytes()
        )
        db.add(new_face)
        db.commit()
        db.refresh(new_face)


class FaceRecognitionService:
    def __init__(self, detector, preprocessor, embedding_generator, comparator):
        self.detector = detector
        self.preprocessor = preprocessor
        self.embedding_generator = embedding_generator
        self.comparator = comparator

    def _load_and_convert_image(self, image_file: UploadFile) -> np.ndarray:
        """Load and convert image to RGB format"""
        contents = image_file.file.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return img_array
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) if len(img_array.shape) == 2 else cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    def _extract_face(self, image: np.ndarray, face_box: dict) -> np.ndarray:
        """Extract face region from image"""
        x, y, width, height = face_box['box']
        x, y = max(0, x), max(0, y)
        return image[y:y+height, x:x+width]

    def _process_single_face(self, face_img: np.ndarray, face_db: list) -> dict:
        """Process a single face and return recognition results"""
        processed_face = self.preprocessor(face_img)
        face_embedding = self.embedding_generator(processed_face)
        identity, confidence = self.comparator(face_embedding, 0.6, face_db)
        return {
            'identity': identity,
            'confidence': confidence
        }

    async def recognize(self, image: UploadFile, db: Session):
        """Main recognition method"""
        try:
            # Load and convert image
            img_array_rgb = self._load_and_convert_image(image)

            # Detect faces
            faces = self.detector.detect_faces(img_array_rgb)
            if not faces:
                return JSONResponse(content={'face': "Not Found"})

            # Get face database
            face_db = db.query(Face).all()

            # Process each detected face
            results = []
            for face in faces:
                face_img = self._extract_face(img_array_rgb, face)
                result = self._process_single_face(face_img, face_db)
                results.append(result)

            return {
                'face': "Found",
                'faces': results
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )