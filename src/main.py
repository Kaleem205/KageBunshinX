import cv2
from recognition.vision_engine import VisionEngine

def main():
    print("Initializing KageBunshinX Vision Engine...")
    engine = VisionEngine()
    
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0

    print("Engine ready. Press 'ESC' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_timestamp_ms += 33 

        user_mask, hand_landmarks = engine.process(frame, frame_timestamp_ms)

        if hand_landmarks and len(hand_landmarks) == 2:
            cv2.putText(frame, "TWO HANDS DETECTED", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "WAITING FOR HANDS...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('KageBunshinX - Engine Test', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
