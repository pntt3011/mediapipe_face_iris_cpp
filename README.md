# Face + Iris Landmarks Real-time Detection in C++ (OpenCV + Tensorflow Lite)

## (Note: This guide is for Windows OS, but the code should work fine on other OS, too)

This project runs on Mediapipe TFLite models without using Mediapipe framework. It can run at **90+ FPS** on **CPU**. 
I perform the test on an AMD Ryzen 7 3700U Pro and the app takes about 5% CPU while running.
For more information:
* Face detection: https://google.github.io/mediapipe/solutions/face_detection.html
* Face landmarks: https://google.github.io/mediapipe/solutions/face_mesh.html
* Iris landmarks: https://google.github.io/mediapipe/solutions/iris.html

## :warning: Why not using GPU ?
Because Tensorflow Lite only supports GPU delegate for Android and IOS.
For more information: https://www.tensorflow.org/lite/performance/gpu

## :computer: Requirements:

### Hardware: Windows 10 64-bit

### Visual Studio 2019

### CMake >= 3.16
You can follow instructions at https://www.40tude.fr/compile-cpp-code-with-vscode-cmake-nmake/

### OpenCV (for Demo)
<details>
  <summary>How to install (Windows 64-bit)</summary>

1. Download and install pre-built binaries at https://sourceforge.net/projects/opencvlibrary/files/4.5.3/opencv-4.5.3-vc14_vc15.exe/download  
2. Add `<opencv-install-folder>/build/x64/vc15/bin` and `<opencv-install-folder>/build/x64/vc15/lib` to PATH.
</details>
Since the prebuilt OPENCV libraries do not contain the 32-bit version, you will have to manually build it using cmake.
https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html
  
### Tensorflow Lite
<details>
  <summary>How to use pre-built library</summary>

1. Download and extract tensorflowlite.zip from https://github.com/shigure3011/mediapipe_face_iris_cpp/releases
2. Change `TFLite_PATH` in CMakeLists.txt
3. Add `TFLite_LIBS` to PATH 

</details>

## :key: How to use:
1. Clone this repo and go to FaceMeshCpp folder
2. Run `cmake -S . -B build`
3. Run `cmake --build build --config Release --target FaceMeshCpp`
4. Now it will build an `.exe` at `~/build/Release`. Make sure to copy `model` folder to `~/build/Release/` before running.
