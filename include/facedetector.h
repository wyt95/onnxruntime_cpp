#ifndef _FACE_DETECTOR_H_
#define _FACE_DETECTOR_H_
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <dirent.h>
#include <unistd.h>

using namespace std;
using namespace cv;

#define YKX_SUCCESS     0
#define YKX_FAILED      1
#define YKX_PARAM_ERROR 2

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    
    for (size_t i = 0; i != input.size(); ++i) {
        /*sum += */y[i] = std::exp(input[i] /*- rowmax*/);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < input.size()/2; ++i) {
        sum = y[i] + y[i + input.size()/2];
        input[i] = y[i] / sum;
        input[i + input.size()/2] = y[ i + input.size()/2] / sum;
    }
}

class FaceDetector
{
public:
    enum COLOR_ORDER{
        GRAY,
        RGBA,
        RGB,
        BGRA,
        BGR
    };
    enum MODEL_VERSION{
        MODEL_PNET,
        MODEL_RNET,
        MODEL_ONET
    };
    enum NMS_TYPE{
        MIN,
        UNION,
    };
    enum IMAGE_DIRECTION{
        ORIENT_LEFT,
        ORIENT_RIGHT,
        ORIENT_UP,
        ORIENT_DOWN,
    };
    struct BoundingBox{
        //rect two points
        float x1, y1;
        float x2, y2;
        //regression
        float dx1, dy1;
        float dx2, dy2;
        //padding stuff
        int px1, py1;
        int px2, py2;
        //cls
        float score;
        //inner points
        float points_x[5];
        float points_y[5];
    };
 
    struct CmpBoundingBox{
        bool operator() (const BoundingBox& b1, const BoundingBox& b2)
        {
            return b1.score > b2.score;
        }
    };

private:
    double                           img_mean;
    double                           img_var;
    cv::Size                         input_geometry_;
    int                              num_channels_;
    MODEL_VERSION                    model_version;

private:
    Ort::SessionOptions session_option;
    //Ort::MemoryInfo     memory_info;
    std::unique_ptr<Ort::Session>    session_PNet;
    std::unique_ptr<Ort::Session>    session_RNet;
    std::unique_ptr<Ort::Session>    session_ONet;
    std::unique_ptr<Ort::Env>        m_OrtEnv;


    //model dir
    std::vector<string> m_model_dir;
    std::string      m_pModel_dir;
    std::string      m_rModel_dir;
    std::string      m_oModel_dir;

    //PNet
    vector<const char*>         m_PNetInputNodeNames;
    vector<vector<int64_t> >    m_PNetInputNodesDims;

    vector<const char*>         m_PNetOutputNodeNames;
    vector<vector<int64_t> >    m_PNetOutputNodesDims;

    //RNet
    vector<const char*>         m_RNetInputNodeNames;
    vector<vector<int64_t> >    m_RNetInputNodesDims;

    vector<const char*>         m_RNetOutputNodeNames;
    vector<vector<int64_t> >    m_RNetOutputNodesDims;

    //ONet
    vector<const char*>         m_ONetInputNodeNames;
    vector<vector<int64_t> >    m_ONetInputNodesDims;

    vector<const char*>         m_ONetOutputNodeNames;
    vector<vector<int64_t> >    m_ONetOutputNodesDims;

    std::mutex m_onnx_mutex;

public:
    FaceDetector(std::vector<string> model_dir);

    ~FaceDetector();
    
    vector< BoundingBox > Detect (const cv::Mat& img, const COLOR_ORDER color_order, const IMAGE_DIRECTION orient, \
                                    int min_size = 20, float P_thres = 0.6, float R_thres = 0.7, float O_thres =0.7,\
                                         bool is_fast_resize = true, float scale_factor = 0.709);

    //Get model I/O info.
    int64_t GetOnnxModelInfo(std::vector<string> model_dir);

    void GetOnnxModelInputInfo(Ort::Session &session_net, 
                               std::vector<const char*> &input_node_names, 
                               vector<vector<int64_t> > &input_node_dims, 
                               std::vector<const char*> &output_node_names, 
                               vector<vector<int64_t> > &output_node_dims);

    void Init();

private:
    void generateBoundingBox(const vector<float>& boxRegs, const vector<int64_t>& box_shape,
                             const vector<float>& cls, const vector<int64_t>& cls_shape,
                             float scale, float threshold, vector<BoundingBox>& filterOutBoxes
                            );

    void filteroutBoundingBox(const vector<BoundingBox>& boxes, 
                              const vector<float>& boxRegs, const vector<int64_t>& box_shape,
                              const vector<float>& cls, const vector<int64_t>& cls_shape,
                              const vector< float >& points, const vector< int64_t >& points_shape,
                              float threshold, vector<BoundingBox>& filterOutBoxes);

    float iou(BoundingBox box1, BoundingBox box2, NMS_TYPE type);
    vector<BoundingBox> nms(std::vector<BoundingBox>& vec_boxs, float threshold, NMS_TYPE type);

    void Padding(vector<BoundingBox> &totalBoxes, int img_w, int img_h);

    void copy_one_patch(const cv::Mat& img, BoundingBox& input_box, float *data_to, cv::Size target_size, int idx, const char *p_str);
};

#endif