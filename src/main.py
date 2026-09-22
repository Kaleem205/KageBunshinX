import cv2
from recognition.vision_engine import VisionEngine
from fsm.jutsu_fsm import JutsuFSM

def main():
    print("Initializing KageBunshinX Vision Engine...")
    engine = VisionEngine()
    
    # Initialize the State Machine
    fsm = JutsuFSM(activation_distance=0.08, cooldown_max=15)
    
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0

    print("Engine ready. Press 'ESC' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_timestamp_ms += 33 

        # 1. Get raw AI data
        user_mask, hand_landmarks = engine.process(frame, frame_timestamp_ms)

        # 2. Feed data to the State Machine
        current_state = fsm.update(hand_landmarks)

        # 3. Display the current state on screen
        if current_state == "ACTIVE":
            cv2.putText(frame, "JUTSU ACTIVE!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        else:
            cv2.putText(frame, "IDLE", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        cv2.imshow('KageBunshinX - FSM Test', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()