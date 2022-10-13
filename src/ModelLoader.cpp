#include "ModelLoader.hpp"

#include <iostream>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

#define INPUT_NORM_MEAN 127.5f
#define INPUT_NORM_STD  127.5f


my::ModelLoader::ModelLoader(std::string modelPath) {
    loadModel(modelPath.c_str());
    buildInterpreter();
    allocateTensors();
    fillInputTensors();
    fillOutputTensors();

    m_inputLoads.resize(getNumberOfInputs(), false);
}


std::vector<int> my::ModelLoader::getInputShape(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].dims;

    return std::vector<int>();
}


float* my::ModelLoader::getInputData(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].data;

    return nullptr;
}


size_t my::ModelLoader::getInputSize(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].bytes;

    return 0;
}


int my::ModelLoader::getNumberOfInputs() const {
    return m_inputs.size();
}


std::vector<int> my::ModelLoader::getOutputShape(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].dims;
        
    return std::vector<int>();
}


float* my::ModelLoader::getOutputData(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].data;

    return nullptr;
}


size_t my::ModelLoader::getOutputSize(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].bytes;

    return 0;
}


int my::ModelLoader::getNumberOfOutputs() const {
    return m_outputs.size();
}


void my::ModelLoader::loadImageToInput(const cv::Mat& inputImage, int idx) {
    if (isIndexValid(idx, 'i')) {
        cv::Mat resizedImage = preprocessImage(inputImage, idx); // Need optimize
        loadBytesToInput(resizedImage.data, idx);
    }
}


void my::ModelLoader::loadBytesToInput(const void* data, int idx) {
    if (isIndexValid(idx, 'i')) {
        memcpy(m_inputs[idx].data, data, m_inputs[idx].bytes);
        m_inputLoads[idx] = true;
    }
}


void my::ModelLoader::runInference() {
    inputChecker();
    m_interpreter->Invoke(); // Tflite inference
}


std::vector<float> my::ModelLoader::loadOutput(int index) const {
    if (isIndexValid(index, 'o')) {
        int sizeInByte = m_outputs[index].bytes;
        int sizeInFloat = sizeInByte / sizeof(float);

        std::vector<float> inference(sizeInFloat);
        memcpy(&(inference[0]), m_outputs[index].data, sizeInByte);
        
        return inference;
    }
    return std::vector<float>();
}


//-------------------Private methods start here-------------------

void my::ModelLoader::loadModel(const char* modelPath) {
    m_model = tflite::FlatBufferModel::BuildFromFile(modelPath);
    if (m_model == nullptr) {
        std::cerr << "Fail to build FlatBufferModel from file: " << modelPath << std::endl;
        std::exit(1);
    }  
}


void my::ModelLoader::buildInterpreter(int numThreads) {
    tflite::ops::builtin::BuiltinOpResolver resolver;

    if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
        std::exit(1);
    }
    m_interpreter->SetNumThreads(numThreads);
}


void my::ModelLoader::allocateTensors() {
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        std::exit(1);
    }
}


void my::ModelLoader::fillInputTensors() {
    for (auto input: m_interpreter->inputs()) {
        TfLiteTensor* inputTensor =  m_interpreter->tensor(input);
        TfLiteIntArray* dims =  inputTensor->dims;

        m_inputs.push_back({
            inputTensor->data.f,
            inputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}


void my::ModelLoader::fillOutputTensors() {
    for (auto output: m_interpreter->outputs()) {
        TfLiteTensor* outputTensor =  m_interpreter->tensor(output);
        TfLiteIntArray* dims =  outputTensor->dims;

        m_outputs.push_back({
            outputTensor->data.f,
            outputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}


bool my::ModelLoader::isIndexValid(int idx, const char c) const {
    int size = 0;
    if (c == 'i')
        size = m_inputs.size();
    else if (c == 'o')
        size = m_outputs.size();
    else 
        return false;

    if (idx < 0 || idx >= size) {
        std::cerr << "Index " << idx << " is out of range (" \
        << size << ")." << std::endl;
        return false;
    }
    return true;
}


bool my::ModelLoader::isAllInputsLoaded() const {
    return (
        std::find(m_inputLoads.begin(), m_inputLoads.end(), false)
     == m_inputLoads.end()); 
}


void my::ModelLoader::inputChecker() {
    if (isAllInputsLoaded() == false) {
        std::cerr << "Input ";
        for (int i = 0; i < m_inputLoads.size(); ++i) {
            if (m_inputLoads[i] == false) {
                std::cerr << i << " ";
            }
        }
        std::cerr << "haven't been loaded." << std::endl;
        std::exit(1);
    }
    std::fill(m_inputLoads.begin(), m_inputLoads.end(), false);
}


cv::Mat my::ModelLoader::preprocessImage(const cv::Mat& in, int idx) const {
    auto out = convertToRGB(in);

    std::vector<int> inputShape = getInputShape(idx);
    int H = inputShape[1];
    int W = inputShape[2]; 

    cv::Size wantedSize = cv::Size(W, H);
    cv::resize(out, out, wantedSize);

    /*
    Equivalent to (out - mean)/ std
    */
    out.convertTo(out, CV_32FC3, 1 / INPUT_NORM_STD, -INPUT_NORM_MEAN / INPUT_NORM_STD);
    return out;
}


cv::Mat my::ModelLoader::convertToRGB(const cv::Mat& in) const {
    cv::Mat out;
    int type = in.type();

    if (type == CV_8UC3) {
        cv::cvtColor(in, out, cv::COLOR_BGR2RGB);
    }
    else if (type == CV_8UC4) {
        cv::cvtColor(in, out, cv::COLOR_BGRA2RGB);
    }
    else {
        std::cerr << "Image of type " << type << " not supported" << std::endl;
        std::exit(1);
    }
    return out;
}