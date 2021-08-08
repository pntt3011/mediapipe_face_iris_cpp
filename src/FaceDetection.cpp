#include "FaceDetection.hpp"


my::FaceDetection::FaceDetection(std::string modelDir) :
    my::ModelLoader(modelDir + std::string("/face_detection_short.tflite")) 
{}


void my::FaceDetection::loadImageToInput(const cv::Mat& in, int index) {
    m_originImage = in;
    ModelLoader::loadImageToInput(in);
}


void my::FaceDetection::runInference() {
    ModelLoader::runInference();

    auto regressor = getFaceRegressor();
    auto classificator = getFaceClassificator();
    auto detection = m_postProcessor.getHighestScoreDetection(regressor, classificator);

    if (detection.classId != -1) {
        /*
        The detection is still in local shape [0..1]
        */
        m_roi = calculateRoiFromDetection(detection);
    }
    else {
        m_roi = cv::Rect();
    }
}


cv::Mat my::FaceDetection::getOriginalImage() const {
    return m_originImage;
}


std::vector<float> my::FaceDetection::getFaceRegressor() const {
    return ModelLoader::loadOutput(0);
}


std::vector<float> my::FaceDetection::getFaceClassificator() const {
    return ModelLoader::loadOutput(1);
}


cv::Rect my::FaceDetection::getFaceRoi() const {
    return m_roi;
}


cv::Mat my::FaceDetection::cropFrame(const cv::Rect& roi) const {
    cv::Mat frame = getOriginalImage();
    cv::Size originalSize(roi.size());

    cv::Point offsetStart(0, 0);
    cv::Point offsetEnd(roi.width, roi.height);

    /*
    Padding the frame with 0 if the Roi is out of frame size.
    */
    auto pt1 = roi.tl();
    auto pt2 = roi.br();

    if (pt1.x < 0) {
        offsetStart.x -= pt1.x; pt1.x = 0;
    }
    if (pt1.y < 0) {
        offsetStart.y -= pt1.y; pt1.y = 0;
    }
    if (pt2.x >= frame.cols) {
        offsetEnd.x -= pt2.x - frame.cols + 1; 
        pt2.x = frame.cols - 1;
    }
    if (pt2.y >= frame.rows) {
        offsetEnd.y -= pt2.y - frame.rows + 1; 
        pt2.y = frame.rows - 1;
    }

    cv::Mat face(originalSize, CV_8UC3, cv::Scalar(0));
    frame(cv::Rect(pt1, pt2)).copyTo(face(cv::Rect(offsetStart, offsetEnd)));
    return face;
}

//-------------------Private methods start here-------------------

cv::Rect my::FaceDetection::calculateRoiFromDetection(const Detection& detection) const {
    int origWidth = m_originImage.size().width;
    int origHeight = m_originImage.size().height;
    
    auto center = (detection.roi.tl() + detection.roi.br()) * 0.5f;
    center.x *= origWidth;
    center.y *= origHeight;

    auto w = detection.roi.width * origWidth * 1.5f;
    auto h = detection.roi.height * origHeight * 2.f;

    return cv::Rect((int)center.x - w/2, (int)center.y - h/2, (int)w, (int)h);
}