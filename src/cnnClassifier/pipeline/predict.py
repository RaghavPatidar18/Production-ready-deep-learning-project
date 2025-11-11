import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import os
import json
from pathlib import Path

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.class_mapping = self._load_class_indices()

    def _load_class_indices(self):
        """Load class indices from training artifacts or use default mapping"""
        class_indices_path = Path("artifacts/training/class_indices.json")
        
        if class_indices_path.exists():
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            # Reverse the mapping: class_name -> index becomes index -> class_name
            index_to_class = {v: k for k, v in class_indices.items()}
            print(f"Loaded class mapping from file: {index_to_class}")
            return index_to_class
        else:
            # Fallback to default mapping based on alphabetical order
            print("Warning: class_indices.json not found. Using default mapping.")
            return {
                0: 'freshripe',
                1: 'freshunripe',
                2: 'overripe',
                3: 'ripe',
                4: 'rotten',
                5: 'unripe'
            }

    
    def predict(self):
        model = load_model(os.path.join("artifacts", "training", "model.h5"))
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Rescale to match training
        test_image = np.expand_dims(test_image, axis=0)
        
        # Get prediction probabilities for all classes
        predictions = model.predict(test_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        
        # Map index to class name
        predicted_class = self.class_mapping[predicted_class_idx]
        
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"All probabilities: {predictions[0]}")
        
        return [{
            "image": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }]
