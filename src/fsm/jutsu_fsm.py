import math

class JutsuFSM:
    def __init__(self, activation_distance=0.08, cooldown_max=15):
        self.state = "IDLE"
        self.cooldown_frames = 0
        self.activation_distance = activation_distance
        self.cooldown_max = cooldown_max

    def update(self, hand_landmarks):
        seal_formed = False
        
        # Check if exactly two hands are present
        if hand_landmarks and len(hand_landmarks) == 2:
            # Landmark 8 is the tip of the index finger
            idx1 = hand_landmarks[0].landmark[8]
            idx2 = hand_landmarks[1].landmark[8]
            
            # Calculate 2D Euclidean distance between the fingertips
            dist = math.dist([idx1.x, idx1.y], [idx2.x, idx2.y])
            
            if dist < self.activation_distance:
                seal_formed = True

        # State Machine Logic
        if seal_formed:
            self.state = "ACTIVE"
            self.cooldown_frames = self.cooldown_max
        elif self.cooldown_frames > 0:
            self.cooldown_frames -= 1
        else:
            self.state = "IDLE"
            
        return self.state