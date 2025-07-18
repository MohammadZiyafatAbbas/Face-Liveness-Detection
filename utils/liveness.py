import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import skimage.feature
import scipy.stats
import time

class LivenessDetector:
    def __init__(self, landmarks_path=None):
        # Determine the path to the landmarks file
        if landmarks_path is None:
            # Default locations to check
            possible_paths = [
                "shape_predictor_68_face_landmarks.dat",
                os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")
            ]
            
            # Find the first existing file
            landmarks_path = next((path for path in possible_paths if os.path.exists(path)), None)
        
        # Validate file exists
        if not landmarks_path or not os.path.exists(landmarks_path):
            raise FileNotFoundError(f"Shape predictor file not found. Please download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and place in the project directory.")
        
        # Initialize detector and predictor
        self.eye_ar_thresh = 0.19  # Balanced value
        self.eye_ar_consec_frames = 2
        self.detector = dlib.get_frontal_face_detector()
        
        try:
            self.predictor = dlib.shape_predictor(landmarks_path)
        except RuntimeError as e:
            print(f"Error loading shape predictor: {e}")
            raise
        
        # Balanced texture analysis parameters
        self.texture_threshold_min = 0.3  # Increased from 0.05
        self.texture_threshold_max = 5.0  # Slightly reduced
        
        # Eye blink tracking
        self.eye_counter = 0
        self.blink_detected = False
        self.last_blink_time = time.time()
        self.blink_timeout = 6.0  # Balanced timeout
        
        # Movement tracking - critical for detecting photos vs real faces
        self.prev_landmarks = None
        self.movement_detected = False
        self.movement_threshold = 1.8  # Increased slightly to ignore small movements in photos
        self.last_movement_time = time.time()
        self.movement_timeout = 3.5  # Balanced timeout
        
        # Require more consistent movement for liveness
        self.movement_history = []
        self.movement_history_size = 10
        self.min_movement_count = 3  # Require at least 3 detected movements in history
        
        # Liveness history for temporal consistency
        self.liveness_history = []
        self.history_size = 12  # Balanced history size
        self.min_positive_ratio = 0.5  # Balanced ratio
        
        # Initialization flag - skip first few frames for calibration
        self.frame_counter = 0
        self.calibration_frames = 10
    
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_head_movement(self, landmarks, prev_landmarks=None):
        nose_tip = landmarks[30]
        
        # Check movement if we have previous landmarks
        if prev_landmarks is not None:
            prev_nose = prev_landmarks[30]
            movement = distance.euclidean(nose_tip, prev_nose)
            
            current_time = time.time()
            # Check if we should reset movement detection
            if current_time - self.last_movement_time > self.movement_timeout:
                self.movement_detected = False
            
            # Movement detected
            if movement > self.movement_threshold:
                self.movement_detected = True
                self.last_movement_time = current_time
                
                # Add to movement history
                self.movement_history.append(1)
            else:
                # No movement
                self.movement_history.append(0)
                
            # Keep history at fixed size
            if len(self.movement_history) > self.movement_history_size:
                self.movement_history.pop(0)
        
        # Determine position
        if nose_tip[0] < 200:  
            return "Left"
        elif nose_tip[0] > 400: 
            return "Right"
        else:
            return "Center"

    def detect_mouth_movement(self, landmarks, prev_landmarks=None):
        mouth = landmarks[48:68]
        upper_lip = mouth[13]  
        lower_lip = mouth[19]  
        mouth_open = distance.euclidean(upper_lip, lower_lip)
        
        mouth_movement = False
        if prev_landmarks is not None:
            prev_mouth = prev_landmarks[48:68]
            prev_upper_lip = prev_mouth[13]
            prev_lower_lip = prev_mouth[19]
            prev_mouth_open = distance.euclidean(prev_upper_lip, prev_lower_lip)
            
            # Detect significant change in mouth openness
            if abs(mouth_open - prev_mouth_open) > 2.5:  # Balanced threshold
                mouth_movement = True
        
        if mouth_open > 19:  # Balanced threshold
            return "Open", mouth_movement
        else:
            return "Closed", mouth_movement
    
    def calculate_entropy(self, image):
        """
        Calculate entropy using scipy
        """
        # Flatten the image and calculate histogram
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        
        # Normalize histogram
        prob_dist = hist / hist.sum()
        
        # Remove zero probabilities to avoid log(0)
        prob_dist = prob_dist[prob_dist > 0]
        
        # Calculate entropy
        return -np.sum(prob_dist * np.log2(prob_dist))
    
    def analyze_texture(self, face_region):
        """
        Perform texture analysis using multiple techniques
        """
        # Validate face region
        if face_region is None or face_region.size == 0:
            print("Empty face region detected")
            return 0.0
        
        # Convert to grayscale
        try:
            if len(face_region.shape) > 2:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
        except Exception as e:
            print(f"Error converting face region to grayscale: {e}")
            return 0.0
        
        # 1. Local Binary Patterns (LBP)
        try:
            lbp = skimage.feature.local_binary_pattern(
                gray, 
                P=8,  # Number of circular neighboring points 
                R=1   # Radius of circle
            )
        except Exception as e:
            print(f"Error computing LBP: {e}")
            lbp = gray  # Fallback
        
        # 2. Pixel Intensity Variation
        pixel_std = np.std(gray)
        
        # 3. Entropy Analysis
        entropy = self.calculate_entropy(gray)
        
        # Combine features - adjusted weights
        texture_score = (
            np.std(lbp) * 0.45 +  # Balanced weight
            pixel_std * 0.85 +    # Balanced weight
            entropy / 9           # Balanced weight
        )
        
        return texture_score
    
    def detect_blink(self, ear):
        """
        Track eye blinks over consecutive frames
        """
        current_time = time.time()
        
        # Reset blink detection after timeout
        if current_time - self.last_blink_time > self.blink_timeout:
            self.blink_detected = False
        
        if ear < self.eye_ar_thresh:
            self.eye_counter += 1
        else:
            # Check if we've had enough consecutive frames with closed eyes to count as a blink
            if self.eye_counter >= self.eye_ar_consec_frames:
                self.blink_detected = True
                self.last_blink_time = current_time
            
            # Reset counter
            self.eye_counter = 0
            
        return self.blink_detected
    
    def check_pixel_variation(self, face_region):
        """
        Check for pixel variation over small regions - printed faces tend to have uniform patterns
        """
        if face_region is None or face_region.size == 0:
            return 0.0
            
        # Convert to grayscale
        try:
            if len(face_region.shape) > 2:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
        except Exception as e:
            print(f"Error in check_pixel_variation: {e}")
            return 0.0
        
        # Divide the face into a 5x5 grid and compute variation in each cell
        h, w = gray.shape
        cell_h, cell_w = h // 5, w // 5
        variations = []
        
        for i in range(5):
            for j in range(5):
                # Get cell coordinates
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                # Ensure we're within bounds
                y2 = min(y2, h)
                x2 = min(x2, w)
                
                # Extract cell and compute standard deviation
                if y1 < y2 and x1 < x2:
                    cell = gray[y1:y2, x1:x2]
                    variations.append(np.std(cell))
        
        # Calculate the standard deviation of variations across cells
        # High value = high variation between different parts of the face = more likely real
        if variations:
            return np.std(variations)
        else:
            return 0.0
            
    def check_glare_reflections(self, face_region):
        """
        Check for glare or reflections that might indicate a screen or printed photo
        """
        if face_region is None or face_region.size == 0:
            return 0
            
        # Convert to grayscale if needed
        try:
            if len(face_region.shape) > 2:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
        except Exception as e:
            print(f"Error in check_glare_reflections: {e}")
            return 0
            
        # Look for very bright pixels (potential glare)
        # Count pixels that are very bright (top 5% of brightness range)
        threshold = 245  # Very bright pixels
        bright_pixels = np.sum(gray > threshold)
        
        # Calculate percentage of bright pixels
        total_pixels = gray.size
        bright_percentage = (bright_pixels / total_pixels) * 100
        
        return bright_percentage
    
    def detect(self, frame):
        # Validate input frame
        if frame is None or frame.size == 0:
            print("Empty frame received")
            return frame, False
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Skip first few frames for calibration
        if self.frame_counter < self.calibration_frames:
            cv2.putText(frame, "Calibrating...", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            return frame, False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        is_live = False
        
        if len(faces) == 0:
            # No faces detected
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame, False
        
        for face in faces:
            # Extract face region
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            
            # Ensure face region is within frame bounds
            if (x < 0 or y < 0 or 
                x + w > frame.shape[1] or 
                y + h > frame.shape[0]):
                print("Face region out of frame bounds")
                continue
            
            try:
                face_region = frame[y:y+h, x:x+w]
            except Exception as e:
                print(f"Error extracting face region: {e}")
                continue
            
            # Perform landmark detection
            try:
                landmarks = self.predictor(gray, face)
                landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            except Exception as e:
                print(f"Landmark detection error: {e}")
                continue
            
            # Eye analysis
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Check for blinks
            blink_detected = self.detect_blink(ear)
            
            # Head and mouth movement
            head_direction = self.detect_head_movement(landmarks, self.prev_landmarks)
            mouth_state, mouth_moved = self.detect_mouth_movement(landmarks, self.prev_landmarks)
            
            # Texture analysis
            texture_score = self.analyze_texture(face_region)
            
            # Additional micro-texture variation check
            pixel_variation = self.check_pixel_variation(face_region)
            
            # Check for glare/reflections (potential indicator of screen or printed photo)
            glare_percentage = self.check_glare_reflections(face_region)
            
            # Store current landmarks for next frame comparison
            self.prev_landmarks = landmarks
            
            # Liveness score calculation - BALANCED LOGIC
            # 1. Texture in expected range
            valid_texture = self.texture_threshold_min <= texture_score <= self.texture_threshold_max
            
            # 2. Check for eye activity (open eyes or blinking)
            eye_activity = ear > self.eye_ar_thresh or blink_detected
            
            # 3. Check for significant movement over time (important for detecting photos)
            movement_count = sum(self.movement_history)
            consistent_movement = movement_count >= self.min_movement_count
            
            # 4. Micro-texture variation
            valid_variation = pixel_variation > 0.65  # Balanced threshold
            
            # 5. Glare check - high glare might indicate printed photo or screen
            low_glare = glare_percentage < 2.0  # Less than 2% bright pixels
            
            # ID cards often fail the movement check but pass texture checks
            # Real faces should pass movement and at least one other check
            
            # Calculate criteria score
            criteria_met = 0
            if valid_texture:
                criteria_met += 1
            if eye_activity:
                criteria_met += 1
            if consistent_movement:  # This is critical for detecting photos vs real faces
                criteria_met += 2    # Give movement double weight
            if valid_variation:
                criteria_met += 1
            if low_glare:
                criteria_met += 1
                
            # Initial liveness assessment - need movement plus at least 2 other criteria
            # or at least 4 total criteria
            frame_is_live = (consistent_movement and criteria_met >= 3) or criteria_met >= 4
            
            # Update liveness history
            self.liveness_history.append(frame_is_live)
            if len(self.liveness_history) > self.history_size:
                self.liveness_history.pop(0)
            
            # Overall liveness is determined by recent history
            positive_count = sum(self.liveness_history)
            is_live = positive_count / len(self.liveness_history) >= self.min_positive_ratio
            
            # Display status and debugging info
            status_text = "LIVE" if is_live else "POTENTIAL SPOOF"
            status_color = (0, 255, 0) if is_live else (0, 0, 255)
            
            cv2.putText(frame, status_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
            
            # Visualize metrics
            cv2.putText(frame, f"Head: {head_direction}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Mouth: {mouth_state}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Texture: {texture_score:.2f} ({valid_texture})", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # cv2.putText(frame, f"Micro-var: {pixel_variation:.2f} ({valid_variation})", (10, 150),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Movement: {movement_count}/{self.movement_history_size} ({consistent_movement})", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Glare: {glare_percentage:.2f}% ({low_glare})", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 255), 2)
            cv2.putText(frame, f"Criteria: {criteria_met}/6", (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Live conf: {positive_count}/{len(self.liveness_history)}", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        return frame, is_live
    