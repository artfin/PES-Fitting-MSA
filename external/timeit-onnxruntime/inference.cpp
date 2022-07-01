#include <chrono>
#include <cassert>
#include <cstring>
#include <cmath>
#include <numeric>
#include <random>

#include <iostream>
#include <iomanip>

#include <onnxruntime_cxx_api.h>

// my C implementation [order=4; NPOLY=79]
extern "C" {
    void evpoly(double x[], double p[]);
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


// see https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h 
std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

void prepare_yij(std::vector<double> const& x, std::vector<double> & yij) {
    const double a0 = 2.0;
    const size_t NATOMS = 7;

    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < NATOMS; ++i) {
        for (size_t j = i + 1; j < NATOMS; ++j) {
            if (i == 0 && j == 1) { yij[k] = 0.0; k = k + 1; continue; } // H1 H2
            if (i == 0 && j == 2) { yij[k] = 0.0; k = k + 1; continue; } // H1 H3
            if (i == 0 && j == 3) { yij[k] = 0.0; k = k + 1; continue; } // H1 H4
            if (i == 1 && j == 2) { yij[k] = 0.0; k = k + 1; continue; } // H2 H3
            if (i == 1 && j == 3) { yij[k] = 0.0; k = k + 1; continue; } // H2 H4
            if (i == 2 && j == 3) { yij[k] = 0.0; k = k + 1; continue; } // H3 H4
            if (i == 0 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H1 C
            if (i == 1 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H2 C
            if (i == 2 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H3 C
            if (i == 3 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H4 C
            if (i == 4 && j == 5) { yij[k] = 0.0; k = k + 1; continue; } // N1 N2

            drx = ATOMX(x, i) - ATOMX(x, j);
            dry = ATOMY(x, i) - ATOMY(x, j);
            drz = ATOMZ(x, i) - ATOMZ(x, j);

            yij[k] = std::sqrt(drx*drx + dry*dry + drz*drz);
            yij[k] = std::exp(-yij[k]/a0);
            k++;
        }
    }
}
    
double forward(Ort::Session & session, std::vector<double> const& x, std::vector<int64_t> const& inputDims, std::vector<const char*> const& inputNames, 
               std::vector<const char*> const& outputNames)
/*
 * cartesian coordinates -> energy
 */
{
    static std::vector<double> yij(21);
    static std::vector<double> poly(79); 

    prepare_yij(x, yij);
    evpoly(yij.data(), poly.data());

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  
    Ort::Value input_tensor = Ort::Value::CreateTensor<double>(
        memoryInfo, poly.data(), poly.size(), inputDims.data(), inputDims.size()
    );

    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outputNames.data(), 1);
    assert(output_tensor.size() == 1 && output_tensor.front().IsTensor());

    double* energy = output_tensor.front().GetTensorMutableData<double>();

    return *energy;
}

void checkCorrectness(Ort::Session & session, std::vector<int64_t> const& inputDims, std::vector<const char*> const& inputNames, 
                      std::vector<const char*> const& outputNames)
{
    std::vector<double> cc = {
 	  1.1935874160,	  1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	 -1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	  1.1935874160, 	 1.1935874160,
 	  1.1935874160,	 -1.1935874160, 	 1.1935874160,
 	  2.5980762114,   2.5980762114, 	 1.5590762114,
 	  2.5980762114,	  2.5980762114, 	 3.6370762114,
 	  0.0000000000,	  0.0000000000, 	 0.0000000000,
    };
    
    double energy = forward(session, cc, inputDims, inputNames, outputNames);
    double expected = 11484.229412069171; 
    assert(std::abs(energy - expected) < 1e-8);
}

int main()
{
    std::string instanceName{"best-rigid-silu"};
    
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    std::string modelFilepath = "../../models/rigid/best-model/silu.onnx";
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};

    std::vector<double> cc = {
 	  1.1935874160,	  1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	 -1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	  1.1935874160, 	 1.1935874160,
 	  1.1935874160,	 -1.1935874160, 	 1.1935874160,
 	  2.5980762114,   2.5980762114, 	 1.5590762114,
 	  2.5980762114,	  2.5980762114, 	 3.6370762114,
 	  0.0000000000,	  0.0000000000, 	 0.0000000000,
    };
    
    double energy = forward(session, cc, inputDims, inputNames, outputNames);

    checkCorrectness(session, inputDims, inputNames, outputNames);

    /*
     * timeit
     */

    std::random_device rd;
    std::mt19937 mt(rd()); 
    std::uniform_real_distribution<> dist(0, 1); 
    
    size_t ncycles = 10000000;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;

    for (size_t k = 0; k < ncycles; ++k) {
        cc[12] = 3.0 + dist(mt); cc[13] = 3.0 + dist(mt); cc[14] = 3.0 + dist(mt);
        cc[15] = 3.0 + dist(mt); cc[16] = 3.0 + dist(mt); cc[17] = 3.0 + dist(mt);

        start = std::chrono::high_resolution_clock::now();
        double energy = forward(session, cc, inputDims, inputNames, outputNames); 
        end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    std::cout << std::fixed << std::setprecision(3);    
    std::cout << "Elapsed nanoseconds:   " << elapsed << std::endl;
    std::cout << "Nanoseconds per cycle: " << elapsed / ncycles << std::endl;

    return 0;
}
