#ifndef DETECTIONPOSTPROCESS_H
#define DETECTIONPOSTPROCESS_H

#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include "opencv2/core.hpp"

#define CLASS_ID        0
#define MIN_THRESHOLD   0.75f
#define DETECTION_SIZE  192
#define NUM_BOXES       2944
#define NUM_COORD       18

namespace my {
    struct Detection {
        cv::Rect2f roi;
        float score;
        int classId;

        Detection() : score(), classId(-1), roi() {}
        Detection(float score, int classId, cv::Rect2f roi) :
            score(score), classId(classId), roi(roi) {}
        ~Detection() = default;
    };

    /*
    A helper class converts the output from Mediapipe Face Detection to Face box.
    */
    class DetectionPostProcess {
        public:
            DetectionPostProcess();
            ~DetectionPostProcess() = default;
            Detection getHighestScoreDetection
            (const std::vector<float>& rawBoxes, const std::vector<float>& scores) const;

        private:
            cv::Rect2f decodeBox(const std::vector<float>& rawBoxes, int index) const;

        private:
            std::vector<cv::Rect2f> m_anchors;
    };
}

#endif // DETECTIONPOSTPROCESS_H