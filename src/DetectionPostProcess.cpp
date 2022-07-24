#include "DetectionPostProcess.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

cv::Rect2f convertAnchorVectorToRect(const std::vector<float> &v)
{
    float cx = v[0];
    float cy = v[1];
    float w = v[2];
    float h = v[3];
    return cv::Rect2f(cx - w / 2, cy - h / 2, w, h);
}

std::vector<cv::Rect2f> generateAnchors()
{
    std::vector<cv::Rect2f> anchors;
    std::ifstream file("./models/anchors.csv");
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream ss(line);
            std::string token;
            std::vector<float> anchor;
            while (std::getline(ss, token, ','))
            {
                anchor.push_back(std::stof(token));
            }
            anchors.push_back(convertAnchorVectorToRect(anchor));
        }
        file.close();
    }
    return anchors;
}


my::DetectionPostProcess::DetectionPostProcess() :
    m_anchors(generateAnchors()) {}


cv::Rect2f my::DetectionPostProcess::decodeBox
(const std::vector<float>& rawBoxes, int index) const {
    auto anchor = m_anchors[index];
    auto center = (anchor.tl() + anchor.br()) * 0.5;
    
    auto boxOffset = index * NUM_COORD;
    float cx = rawBoxes[boxOffset];
    float cy = rawBoxes[boxOffset + 1];
    float w = rawBoxes[boxOffset + 2];
    float h = rawBoxes[boxOffset + 3];

    cx = cx / DETECTION_SIZE * anchor.width + center.x;
    cy = cy / DETECTION_SIZE * anchor.height + center.y;
    w = w / DETECTION_SIZE * anchor.width;
    h = h / DETECTION_SIZE * anchor.height;

    return cv::Rect2f(cx - w/2, cy - h/2, w, h);
}


my::Detection my::DetectionPostProcess::getHighestScoreDetection
(const std::vector<float>& rawBoxes, const std::vector<float>& scores) const {
    my::Detection detection;
    for (int i = 0; i < NUM_BOXES; i++) {
        if (scores[i] > std::max(MIN_THRESHOLD, detection.score)) {
            auto data = decodeBox(rawBoxes, i);
            detection = my::Detection(scores[i], CLASS_ID, data);
        }
    }
    return detection;
}