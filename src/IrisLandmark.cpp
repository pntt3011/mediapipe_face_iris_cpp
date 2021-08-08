#include "IrisLandmark.hpp"
#include <iostream>
#include <thread>

#define EYE_LANDMARKS 71
#define IRIS_LANDMARKS 5

/*
Helper functions
*/
bool __isEyeIndexValid(int idx) {
    if (idx < 0 || idx >= EYE_LANDMARKS) {
        std::cerr << "Index " << idx << " is out of range (" \
        << EYE_LANDMARKS << ")." << std::endl;
        return false;
    }
    return true;
}


bool __isIrisIndexValid(int idx) {
    if (idx < 0 || idx >= IRIS_LANDMARKS) {
        std::cerr << "Index " << idx << " is out of range (" \
        << IRIS_LANDMARKS << ")." << std::endl;
        return false;
    }
    return true;
}


my::IrisLandmark::IrisLandmark(std::string modelPath):
    FaceLandmark(modelPath),
    m_leftIrisLandmarker(modelPath + std::string("/iris_landmark.tflite")),
    m_rightIrisLandmarker(modelPath + std::string("/iris_landmark.tflite"))
    {}


void my::IrisLandmark::runInference() {
    FaceLandmark::runInference();
    auto roi = FaceDetection::getFaceRoi();
    if (roi.empty()) return;

    std::thread t([this]() {this->runEyeInference(true);});
    runEyeInference(false);
    t.join();
}


cv::Point my::IrisLandmark::getEyeLandmarkAt(int index, bool isLeftEye, bool isIris) const {
    if (__isEyeIndexValid(index)) {
        auto model = isLeftEye ? &m_leftIrisLandmarker: &m_rightIrisLandmarker;
        auto eyeRoi = isLeftEye ? m_leftEyeRoi: m_rightEyeRoi;

        float _x = model->getOutputData(isIris)[index * 3];
        float _y = model->getOutputData(isIris)[index * 3 + 1];

        int x = (int)(_x / model->getInputShape()[2] * eyeRoi.width) + eyeRoi.x;
        int y = (int)(_y / model->getInputShape()[1] * eyeRoi.height) + eyeRoi.y;

        return cv::Point(x,y);
    }
    return cv::Point();
}


std::vector<cv::Point> my::IrisLandmark::getAllEyeLandmarks(bool isLeftEye, bool isIris) const {
    if (my::FaceDetection::getFaceRoi().empty())
        return std::vector<cv::Point>();

    int n = isIris ? IRIS_LANDMARKS : EYE_LANDMARKS;

    std::vector<cv::Point> landmarks(n);
    for (int i = 0; i < n; ++i) {
        landmarks[i] = getEyeLandmarkAt(i, isLeftEye, isIris);
    }
    return landmarks;
}


std::vector<float> my::IrisLandmark::loadOutput(int index, bool isLeftEye) const {
    auto model = isLeftEye ? &m_leftIrisLandmarker: &m_rightIrisLandmarker;
    return model->loadOutput();
}


cv::Rect my::IrisLandmark::getEyeRoi(bool isLeftEye) const {
    return isLeftEye ? m_leftEyeRoi: m_rightEyeRoi;
}

//-------------------Private methods start here-------------------

cv::Rect my::IrisLandmark::calculateEyeRoi(cv::Point leftMoft, cv::Point rightMost) const{
    int cx = (leftMoft.x + rightMost.x) / 2;
    int cy = (leftMoft.y + rightMost.y) / 2; 

    int w = std::abs(leftMoft.x - rightMost.x);
    int h = std::abs(leftMoft.y - rightMost.y);
    w = h = std::max(w, h);
    
    return cv::Rect(cx - w/2, cy - h/2, w, h);
}


void my::IrisLandmark::runEyeInference(bool isLeftEye) {
    int idx1 = isLeftEye ? 446 : 244;
    int idx2 = isLeftEye ? 464 : 226;

    auto roi = isLeftEye ? &m_leftEyeRoi: &m_rightEyeRoi;
    auto model = isLeftEye ? &m_leftIrisLandmarker: &m_rightIrisLandmarker;

    auto pt1 = FaceLandmark::getFaceLandmarkAt(idx1);
    auto pt2 = FaceLandmark::getFaceLandmarkAt(idx2);

    *roi = calculateEyeRoi(pt1, pt2);
    auto eyePatch = FaceDetection::cropFrame(*roi);

    model->loadImageToInput(eyePatch);
    model->runInference();
}