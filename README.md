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
   - Download from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During installation, select:
     - **C++ CMake tools for Windows**
     - **MSVC v142 - VS 2019 C++ x64/x86 build tools**
     - **Windows 10 SDK**
     - **C++/CLI support for v142 build tools**
     - **C++ Modules for v142 build tools**

2. **Reinstall `dlib` and `cmake` using pip:**
```sh
pip install cmake dlib
```

### Fixing OpenCV Camera Issues
If OpenCV cannot access your webcam, try:
```sh
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

### Fixing Flask Debugger Errors
If Flask debugger throws exceptions related to streamed responses, restart Flask with:
```sh
set FLASK_ENV=development  # Windows
export FLASK_ENV=development  # macOS/Linux
python app.py
```


