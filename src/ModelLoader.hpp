#ifndef MODELLOADER_H
#define MODELLOADER_H

#include <vector>
#include <memory>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace my {

    template <class T>
    using Matrix = std::vector<std::vector<T>>;

    /*
    A tensor wrapper to save information of tflite tensors.
    Attributes:
        data: a float pointer to tensor data
        bytes: size of data in bytes
        dims: shape of data tensor
    */
    struct TensorWrapper {
        float* data;
        size_t bytes;
        std::vector<int> dims;

        TensorWrapper(float* t_data, size_t t_bytes, int* t_dims, int t_dimSize): 
            data(t_data), bytes(t_bytes), dims(t_dims, t_dims + t_dimSize) {}
    };

    /*
    A model wrapper to simplify the procedure of using tflite's models.
    This class is non-copyable.
    */
    class ModelLoader {
        public:
            /*
            Constructor from a .tflite file
            Parameters:
                modelPath: path to .tflite
            */
            ModelLoader(std::string modelPath);
            ModelLoader(const ModelLoader& other) = delete;
            ModelLoader& operator=(const ModelLoader& other) = delete;
            virtual ~ModelLoader() = default;

            /*
            Get shape of input tensor at index.
            (Note: A model can have multiple inputs)
            Parameters:
                index: index of input tensor
            */
            std::vector<int> getInputShape(int index = 0) const;

            /*
            Get the pointer to the data of input tensor at index.
            (Note: A model can have multiple inputs)
            Parameters:
                index: index of input tensor
            */
            float* getInputData(int index = 0) const;

            /*
            Get size in bytes of input tensor at index.
            (Note: A model can have multiple inputs)
            Parameters:
                index: index of input tensor
            */
            size_t getInputSize(int index = 0) const;

            /*
            Get number of inputs needed to run inference. 
            */
            int getNumberOfInputs() const;

            /*
            Get shape of output tensor at index.
            (Note: A model can have multiple outputs)
            Parameters:
                index: index of output tensor
            */     
            std::vector<int> getOutputShape(int index = 0) const;

            /*
            Get the pointer to the data of output tensor at index.
            (Note: A model can have multiple outputs)
            Parameters:
                index: index of output tensor
            */
            float* getOutputData(int index = 0) const;

            /*
            Get size in bytes of output tensor at index.
            (Note: A model can have multiple outputs)
            Parameters:
                index: index of output tensor
            */
            size_t getOutputSize(int index = 0) const;

            /*
            Get number of outputs from inference.
            */
            int getNumberOfOutputs() const;

            /*
            Load image (BGR format) to model at index 
            (Note: Only support image of type CV_8UC3 and CV_8UC4)
            */
            virtual void loadImageToInput(const cv::Mat& inputImage, int index = 0);

            /*
            Load byte data to model at index
            */
            virtual void loadBytesToInput(const void* data, int index = 0);

            /*
            Run inference on the inputs.
            Can only run when all input tensors have been loaded.
            */
            virtual void runInference();

            /*
            A vector contains output data at index.
            Its shape is flattened from getOutputShape(index)
            */
            virtual std::vector<float> loadOutput(int index = 0) const;


        private:
            /*
            Constructor helper functions
            */
            void loadModel(const char* modelPath);
            void buildInterpreter(int numThreads = -1);
            void allocateTensors();           
            void fillInputTensors();
            void fillOutputTensors();

            /*
            Check if index is valid for input and output tensor
            */
            bool isIndexValid(int index, const char c = 'i') const;

            /*
            Check if all inputs have been loaded
            */
            bool isAllInputsLoaded() const;

            /*
            Process input loads before run inference
            */
            void inputChecker();

            /*
            Convert image to float and resize to getInputShape(idx)
            */
            cv::Mat preprocessImage(const cv::Mat& in, int idx) const;

            /*
            Convert image of type CV_8UC3 or CV_8UC4 to RGB format
            */
            cv::Mat convertToRGB(const cv::Mat& in) const;


        private:
            /*
            Information of input tensors
            */
            std::vector<TensorWrapper> m_inputs;

            /*
            Information of output tensors
            */
            std::vector<TensorWrapper> m_outputs;

            /*
            TFLite core
            */
            std::unique_ptr<tflite::FlatBufferModel> m_model;

            /*
            TFLite core
            */           
            std::unique_ptr<tflite::Interpreter> m_interpreter;

            /*
            Tracking inputs loaded
            */
            std::vector<bool> m_inputLoads;
    };
};

#endif // MODELLOADER_H