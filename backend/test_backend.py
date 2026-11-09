import sys
import os
import unittest
import numpy as np
from fastapi.testclient import TestClient

# Ensure current dir is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
from services.model_service import SIGNS

class TestSignConnectBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["class_count"], 21)
        self.assertEqual(data["backend"], "tf_keras")
        print("PASS: Health endpoint verified.")

    def test_02_classes_endpoint(self):
        resp = self.client.get("/classes")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["classes"]), 21)
        self.assertEqual(data["classes"], SIGNS)
        print("PASS: Classes endpoint verified.")

    def test_03_valid_prediction(self):
        # 30 frames of 126 features
        dummy_input = np.random.randn(30, 126).tolist()
        resp = self.client.post("/predict", json={"input": dummy_input})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sign", data)
        self.assertIn(data["sign"], SIGNS)
        self.assertIn("confidence", data)
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertEqual(len(data["top_candidates"]), 3)
        self.assertEqual(len(data["all_probabilities"]), 21)
        print(f"PASS: Valid prediction verified. Predicted sign: {data['sign']}, Confidence: {data['confidence']:.4f}")

    def test_04_invalid_shape_validation(self):
        # Incorrect frames count (10 instead of 30)
        invalid_input = np.random.randn(10, 126).tolist()
        resp = self.client.post("/predict", json={"input": invalid_input})
        self.assertIn(resp.status_code, [400, 422, 500])

        # Incorrect features count (30, 50)
        invalid_input2 = np.random.randn(30, 50).tolist()
        resp2 = self.client.post("/predict", json={"input": invalid_input2})
        self.assertIn(resp2.status_code, [400, 422, 500])
        print("PASS: Input shape validation correctly rejects malformed sequences.")

    def test_05_sentence_builder(self):
        # 1. Test Computational NLP Engine
        words = ["TODAY", "I", "SAD"]
        resp_nlp = self.client.post("/sentence/build", json={
            "words": words,
            "mode": "nlp"
        })
        self.assertEqual(resp_nlp.status_code, 200)
        data_nlp = resp_nlp.json()
        self.assertIn("Today, I am sad.", data_nlp["sentence"])
        self.assertEqual(data_nlp["method"], "computational_nlp_engine")

        # Complex gloss sentence
        words2 = ["HELLO", "YOU", "ALRIGHT"]
        resp_nlp2 = self.client.post("/sentence/build", json={"words": words2, "mode": "nlp"})
        self.assertEqual(resp_nlp2.status_code, 200)
        self.assertEqual(resp_nlp2.json()["sentence"], "Hello, you are alright!")

        # 2. Test raw concatenation
        resp_raw = self.client.post("/sentence/build", json={
            "words": words,
            "mode": "raw"
        })
        self.assertEqual(resp_raw.status_code, 200)
        self.assertEqual(resp_raw.json()["sentence"], "TODAY I SAD")

        # 3. Test AI model mode fallback (without key)
        resp_ai = self.client.post("/sentence/build", json={
            "words": words,
            "mode": "ai_model"
        })
        self.assertEqual(resp_ai.status_code, 200)
        self.assertEqual(resp_ai.json()["method"], "local_nlp_fallback")
        print(f"PASS: NLP & AI Sentence Prediction verified. Raw: '{resp_raw.json()['sentence']}' -> NLP: '{data_nlp['sentence']}'")

    def test_06_frontend_static_serving(self):
        # Root index.html
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("SignConnect", resp.text)

        # CSS asset
        resp_css = self.client.get("/static/styles.css")
        self.assertEqual(resp_css.status_code, 200)

        # JS asset
        resp_js = self.client.get("/static/app.js")
        self.assertEqual(resp_js.status_code, 200)
        self.assertIn("extractHandFeatures", resp_js.text)
        print("PASS: Frontend assets served successfully.")

if __name__ == "__main__":
    unittest.main()
