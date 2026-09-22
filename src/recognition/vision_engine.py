import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VisionEngine:
    def __init__(self, segmenter_model='selfie_segmenter.tflite'):
        # 1. Selfie Segmenter using Modern Tasks API (CPU Forced)
        base_options_seg = python.BaseOptions(
            model_asset_path=segmenter_model,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options_seg = vision.ImageSegmenterOptions(
            base_options=base_options_seg,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options_seg)

        # 2. Hand Tracker using Rock-Solid Legacy API
        # This completely avoids the Metal GPU crash on M-series chips
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )

    def process(self, image, timestamp_ms):
        """Processes the frame and returns the user mask and hand landmarks."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- Segmenter Processing (Tasks API format) ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        segmentation_result = self.segmenter.segment_for_video(mp_image, timestamp_ms)
        category_mask = np.squeeze(segmentation_result.category_mask.numpy_view())
        condition = np.stack((category_mask,) * 3, axis=-1) > 0

        # --- Hand Processing (Legacy API format) ---
        hand_result = self.hands.process(image_rgb)
        
        # The main script expects a simple boolean/list check
        detected_hands = []
        if hand_result.multi_hand_landmarks:
            detected_hands = hand_result.multi_hand_landmarks

        return condition, detected_hands