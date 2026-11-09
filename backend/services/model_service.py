import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

SIGNS = [
    "AFTERNOON", "ALRIGHT", "DRESS", "GOOD", "HE",
    "HELLO", "I", "IT", "MONSOON", "MORNING",
    "PLEASE", "RICH", "SAD", "SHE", "SUMMER",
    "T-SHIRT", "TELEPHONE", "THANK YOU", "TODAY", "WINTER", "YOU"
]

class SignModelService:
    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        
        self.model_dir = model_dir
        self.h5_path = os.path.join(model_dir, "sign_transformer.h5")
        self.tflite_path = os.path.join(model_dir, "sign_transformer.tflite")
        
        self.model = None
        self.backend_type = None  # 'tf_keras' or 'tflite'
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()

    def _load_model(self):
        # 1. Primary: Load sign_transformer.h5 using tf_keras
        if os.path.exists(self.h5_path):
            try:
                import tf_keras as keras
                logger.info(f"Loading primary Keras model from {self.h5_path}...")
                self.model = keras.models.load_model(self.h5_path)
                self.backend_type = 'tf_keras'
                logger.info("Primary Keras model loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load .h5 with tf_keras: {e}. Attempting fallback...")
        
        # 2. Fallback: Load sign_transformer.tflite
        if os.path.exists(self.tflite_path):
            try:
                from ai_edge_litert.interpreter import Interpreter
                logger.info(f"Loading fallback TFLite model from {self.tflite_path}...")
                self.interpreter = Interpreter(model_path=self.tflite_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.backend_type = 'tflite'
                logger.info("TFLite model loaded successfully.")
                return
            except Exception as e:
                logger.error(f"Failed to load fallback TFLite model: {e}")
                
        raise RuntimeError("Neither sign_transformer.h5 nor sign_transformer.tflite could be loaded.")

    def is_loaded(self) -> bool:
        return self.model is not None or self.interpreter is not None

    def get_classes(self) -> List[str]:
        return SIGNS

    def predict(self, input_sequence: List[Any]) -> Dict[str, Any]:
        """
        Expects input_sequence with shape (30, 126) or (1, 30, 126).
        Returns top prediction, confidence score, and top 3 candidates.
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not initialized.")

        arr = np.array(input_sequence, dtype=np.float32)

        # Handle shape
        if arr.ndim == 2:
            if arr.shape != (30, 126):
                raise ValueError(f"Expected input shape (30, 126), got {arr.shape}")
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            if arr.shape != (1, 30, 126):
                raise ValueError(f"Expected input shape (1, 30, 126), got {arr.shape}")
        else:
            raise ValueError(f"Invalid input dimensions: {arr.ndim}. Expected 2D or 3D array.")

        # Inference
        if self.backend_type == 'tf_keras':
            preds = self.model.predict(arr, verbose=0)[0]
        elif self.backend_type == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], arr)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            raise RuntimeError(f"Unknown backend type: {self.backend_type}")

        # Post-process
        preds = np.array(preds, dtype=np.float32)
        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])
        top_sign = SIGNS[top_idx] if top_idx < len(SIGNS) else f"UNKNOWN_{top_idx}"

        # Top 3 candidates
        top_indices = np.argsort(preds)[::-1][:3]
        candidates = [
            {
                "sign": SIGNS[i] if i < len(SIGNS) else f"UNKNOWN_{i}",
                "confidence": round(float(preds[i]), 4)
            }
            for i in top_indices
        ]

        return {
            "sign": top_sign,
            "confidence": round(top_conf, 4),
            "backend": self.backend_type,
            "top_candidates": candidates,
            "all_probabilities": {
                SIGNS[i]: round(float(preds[i]), 4) for i in range(len(SIGNS))
            }
        }
