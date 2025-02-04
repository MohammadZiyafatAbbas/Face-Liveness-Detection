# Face Liveness Detection  
A real-time face liveness detection system using OpenCV and dlib.

## Features  
- Eye blink detection  
- Head movement detection  
- Mouth movement detection  

## Installation  

1. **Clone the repository**  
   ```sh
   git clone https://github.com/your-username/face-liveness-detection.git
   cd face-liveness-detection
   ```

2. **Create a virtual environment (Recommended)**  
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application  
```sh
python app.py
```
This will start a local Flask server. Open your browser and go to **`http://127.0.0.1:5000/`**.

---

## Troubleshooting  

### 1️⃣ **dlib Installation Issues**
If `dlib` fails to install, try the following:

#### **Windows Users:**  
Install CMake and Visual Studio C++ Build Tools:

```sh
pip install cmake
pip install dlib
```
If this fails, download and install **Visual Studio Build Tools** from:  
🔗 [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Then retry:  
```sh
pip install dlib
```

#### **Linux Users:**  
```sh
sudo apt-get update
sudo apt-get install cmake
pip install dlib
```

### 2️⃣ **OpenCV Not Found Error**  
If `cv2` is not recognized:  
```sh
pip install opencv-python
```

### 3️⃣ **shape_predictor_68_face_landmarks.dat Not Found**  
This file is required for facial landmark detection.  
🔗 Download it from: [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)  
Extract it and place it in the project directory.

---

## Contributors  
- **Your Name** - [GitHub Profile](https://github.com/your-username)  

---

