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

void readFileList(const char* basePath, vector<string>& imgFiles)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    
    if( (dir=opendir(basePath)) == NULL)
    {
        return ;
    }
    
    while( (ptr = readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".") == 0 ||
            strcmp(ptr->d_name, "..") == 0)
            continue;
        else if(ptr->d_type == 8)//file 
        {
            int len = strlen(ptr->d_name);
            if(ptr->d_name[len-1] == 'g' && ptr->d_name[len-2] == 'p' && ptr->d_name[len-3] == 'j')
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if(ptr->d_type == 10)/// link file
        {
            int len = strlen(ptr->d_name);
            if(ptr->d_name[len-1] == 'g' && ptr->d_name[len-2] == 'p' && ptr->d_name[len-3] == 'j')
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if(ptr->d_type == 4)//dir
        {
            memset(base, '\0', sizeof(base));
            strcpy(base, basePath);
            strcat(base, "/");
            strcat(base, ptr->d_name);
            readFileList(base, imgFiles);
        }
    }
    closedir(dir);
}

class FaceDetector
{
public:
    struct FaceInfo
    {

    };

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
    Ort::MemoryInfo     memory_info;
    Ort::Session        session_PNet;
    Ort::Session        session_RNet;
    Ort::Session        session_ONet;

    //model dir
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
    FaceDetector(std::vector<string> model_dir, const MODEL_VERSION model_version);

    ~FaceDetector();
    
    vector< BoundingBox > Detect (const cv::Mat& img, const COLOR_ORDER color_order, const IMAGE_DIRECTION orient, \
                                    int min_size = 20, float P_thres = 0.6, float R_thres = 0.7, float O_thres =0.7,\
                                         bool is_fast_resize = true, float scale_factor = 0.709);

    //获取模型输入输出信息
    int64_t GetOnnxModelInfo(std::vector<string> model_dir);

    void GetOnnxModelInputInfo(Ort::Session& session_net, std::vector<const char*> &input_node_names, std::vector<const char*> &output_node_names);

    void Release();

private:
    void generateBoundingBox(const vector<float>& boxRegs, const vector<int>& box_shape,
                             const vector<float>& cls, const vector<int>& cls_shape,
                             float scale, float threshold, vector<BoundingBox>& filterOutBoxes
                            );
    void filteroutBoundingBox(const vector<BoundingBox>& boxes, 
                              const vector<float>& boxRegs, const vector<int>& box_shape,
                              const vector<float>& cls, const vector<int>& cls_shape,
                              const vector< float >& points, const vector< int >& points_shape,
                              float threshold, vector<BoundingBox>& filterOutBoxes);
    void nms_cpu(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes);

    void pad(vector<BoundingBox>& boxes, int imgW, int imgH);

    void pyrDown(const vector<cv::Mat>& img_channels,float scale, std::vector<cv::Mat>* input_channels);
    void buildInputChannels(const std::vector<cv::Mat>& img_channels, const std::vector<BoundingBox>& boxes,
                            const cv::Size& target_size, std::vector<cv::Mat>* input_channels);
};

#endif