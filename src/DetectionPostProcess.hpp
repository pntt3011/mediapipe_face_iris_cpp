#ifndef DETECTIONPOSTPROCESS_H
#define DETECTIONPOSTPROCESS_H

#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include "opencv2/core.hpp"

#define CLASS_ID        0
#define MIN_THRESHOLD   0.75f
#define DETECTION_SIZE  128
#define NUM_BOXES       896
#define NUM_COORD       16
#define NUM_SIZES       2

namespace my {

    struct AnchorOptions {
        // 2 x 16 x 16 and 6 x 8 x 8 --> 896
        const int sizes[NUM_SIZES] = {16, 8};
        const int numLayers[NUM_SIZES] = {2, 6};

        // The offset for the center of anchors.
        const float offsetX = 0.5f;
        const float offsetY = 0.5f;
    };


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