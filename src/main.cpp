//#include <QCoreApplication>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <unistd.h>
#include <vector>
#include <array>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <string.h>

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
//#include <core/providers/tensorrt/tensorrt_provider_factory.h>

#include <NumCpp.hpp>
#include <typeinfo>

using namespace cv;
using namespace std;

//OrtApi *g;
Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
Ort::SessionOptions session_option{nullptr};

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

void generateBoundingBox(const vector<float>& boxRegs, const vector<int64_t>& box_shape,
                             const vector<float>& cls, const vector<int64_t>& cls_shape,
                             float scale, float threshold, vector<BoundingBox>& filterOutBoxes
                            )
{
    //clear output element
    filterOutBoxes.clear();
    int stride = 2;
    int cellsize = 12;
    assert(box_shape.size() == cls_shape.size());
    assert(box_shape[3] == cls_shape[3] && box_shape[2] == cls_shape[2]);
    assert(box_shape[0] == 1 && cls_shape[0] == 1);
    assert(box_shape[1] == 4 && cls_shape[1] == 2);
    int w = box_shape[3];
    //printf("=====w====%d===\n", w);
    int h = box_shape[2];
    //printf("=====h====%d===\n", h);
    int count = 0;
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            float score = cls[0 * 2 * w * h + 1 * w * h + w * y + x];
            if (score >= 0.6)
            {
                //printf("=====score====%f===\n", score);
                count++;
            }

            if ( score >= threshold)
            {
                BoundingBox box;
                box.dx1 = boxRegs[0 * w * h + w * y + x];
                box.dy1 = boxRegs[1 * w * h + w * y + x];
                box.dx2 = boxRegs[2 * w * h + w * y + x];
                box.dy2 = boxRegs[3 * w * h + w * y + x];
                
                box.x1 = std::floor( (stride * x + 1) / scale );
                box.y1 = std::floor( (stride * y + 1) / scale );
                box.x2 = std::floor( (stride * x + cellsize) / scale );
                box.y2 = std::floor( (stride * y + cellsize) / scale );
                box.score = score;
                //add elements
                filterOutBoxes.push_back(box);
            }
        }
    }

    printf("count:%d\n", count);
}

void filteroutBoundingBox(const vector< BoundingBox >& boxes, 
                                        const vector< float >& boxRegs, const vector< int64_t >& box_shape, 
                                        const vector< float >& cls, const vector< int64_t >& cls_shape, 
                                        const vector< float >& points, const vector< int64_t >& points_shape,
                                        float threshold, vector< BoundingBox >& filterOutBoxes)
{
    filterOutBoxes.clear();

    for(int i = 0; i < boxes.size(); i ++)
    {
        float score = cls[i * 2 + 1];
        //printf("score: %f \n", score);
        if ( score > threshold )
        {
            printf("==%d==score: %f \n", i, score);
            BoundingBox box = boxes[i];
            float w = boxes[i].y2 - boxes[i].y1 + 1;
            float h = boxes[i].x2 - boxes[i].x1 + 1;
            if( points.size() > 0)
            {
                for(int p = 0; p < 5; p ++)
                {
                    box.points_x[p] = points[i * 10 + 5 + p] * w + boxes[i].x1 - 1;
                    box.points_y[p] = points[i * 10 + p] * h + boxes[i].y1 - 1;
                }
            }
            box.dx1 = boxRegs[i * 4 + 0];
            printf("box.dx1:%f\n", box.dx1);
            box.dy1 = boxRegs[i * 4 + 1];
            printf("box.dy1:%f\n", box.dy1);
            box.dx2 = boxRegs[i * 4 + 2];
            printf("box.dx2:%f\n", box.dx2);
            box.dy2 = boxRegs[i * 4 + 3];
            printf("box.dy2:%f\n", box.dy2);

            box.x1 = boxes[i].x1 + box.dy1 * w;
            box.y1 = boxes[i].y1 + box.dx1 * h;
            box.x2 = boxes[i].x2 + box.dy2 * w;
            box.y2 = boxes[i].y2 + box.dx2 * h;

            //rerec
            w = box.x2 - box.x1;
            h = box.y2 - box.y1;
            float l = std::max(w, h);
            box.x1 += (w - l) * 0.5;
            box.y1 += (h - l) * 0.5;
            box.x2 = box.x1 + l;
            box.y2 = box.y1 + l;
            box.score = score;

            filterOutBoxes.push_back(box);
        }
    }
}

enum NMS_TYPE{
        MIN,
        UNION,
    };

struct CmpBoundingBox{
        bool operator() (const BoundingBox& b1, const BoundingBox& b2)
        {
            return b1.score > b2.score;
        }
    };

void nms_cpu(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes)
{
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    //descending sort
    sort(boxes.begin(), boxes.end(), CmpBoundingBox() );
    vector<size_t> idx(boxes.size());
    for(int i = 0; i < idx.size(); i++)
    { 
        idx[i] = i; 
    }
    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);
        //hypothesis : the closer the scores are similar
        vector<size_t> tmp = idx;
        idx.clear();
        for(int i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx].x1, boxes[tmp_i].x1 );
            float inter_y1 = std::max( boxes[good_idx].y1, boxes[tmp_i].y1 );
            float inter_x2 = std::min( boxes[good_idx].x2, boxes[tmp_i].x2 );
            float inter_y2 = std::min( boxes[good_idx].y2, boxes[tmp_i].y2 );
             
            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);
            
            float inter_area = w * h;
            float area_1 = (boxes[good_idx].x2 - boxes[good_idx].x1 + 1) * (boxes[good_idx].y2 - boxes[good_idx].y1 + 1);
            float area_2 = (boxes[i].x2 - boxes[i].x1 + 1) * (boxes[i].y2 - boxes[i].y1 + 1);
            float o = ( type == UNION ? (inter_area / (area_1 + area_2 - inter_area)) : (inter_area / std::min(area_1 , area_2)) );           
            if( o <= threshold )
                idx.push_back(tmp_i);
        }
    }
}


float iou(BoundingBox box1, BoundingBox box2, NMS_TYPE type) 
{ 
    float inter_x1 = std::max(box1.x1, box2.x1); 
    float inter_y1 = std::max(box1.y1, box2.y1); 
    float inter_x2 = std::min(box1.x2, box2.x2); 
    float inter_y2 = std::min(box1.y2, box2.y2); 

    float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
    float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);
    float inter_area = w * h;
    float area_1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    float area_2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
    float iou = ( type == UNION ? (inter_area / (area_1 + area_2 - inter_area)) : (inter_area / std::min(area_1 , area_2)) );

    return iou; 
}

vector<BoundingBox> nms(std::vector<BoundingBox>& vec_boxs, float threshold, NMS_TYPE type) 
{ 
    vector<BoundingBox> results; 
    while(vec_boxs.size() > 0)
    {
        sort(vec_boxs.begin(), vec_boxs.end(), CmpBoundingBox() );
        results.push_back(vec_boxs[0]);
        int index = 1;
        while(index < vec_boxs.size())
        {
            float iou_value = iou(vec_boxs[0], vec_boxs[index], type);
            if(iou_value > threshold)
                vec_boxs.erase(vec_boxs.begin() + index);
            else
                index++;
        }
        vec_boxs.erase(vec_boxs.begin());
    }

    return results;
} 

void GetOnnxModelInputInfo(Ort::Session& session_net, std::vector<const char*> &input_node_names, std::vector<const char*> &output_node_names)
{
    size_t num_input_nodes = session_net.GetInputCount();
    //std::vector<const char*> input_node_names(num_input_nodes);
    input_node_names.resize(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    printf("Number of inputs = %zu\n", num_input_nodes);
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) 
    {
        // print input node names
        char* input_name = session_net.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session_net.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        }

        //allocator.Free(input_name);
    }

    size_t num_output_nodes = session_net.GetOutputCount();
    output_node_names.resize(num_output_nodes);
    std::vector<int64_t> output_node_dims;

    printf("Number of outputs = %zu\n", num_output_nodes);
    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++) 
    {
        // print output node names
        char* output_name = session_net.GetOutputName(i, allocator);
        printf("output %d : name=%s\n", i, output_name);
        output_node_names[i] = output_name;

        // print output node types
        Ort::TypeInfo type_info = session_net.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // print output shapes/dims
        output_node_dims = tensor_info.GetShape();
        printf("output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
        {
            printf("output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
        }

       //allocator.Free(output_name);
    }

}

void copy_one_patch(const cv::Mat& img, BoundingBox& input_box, float *data_to, cv::Size target_size, int i, const char *p_str)
{
    cv::Mat copy_img = img.clone();
    cv::Mat chop_img;

    int height = target_size.height;
    int width = target_size.width;
    float src_height = abs(input_box.px1 - input_box.px2);
    float src_width = abs(input_box.py1 - input_box.py2);
    printf("src_height:%f, src_width:%f\n", src_height, src_width);
    printf("===%d===px1: %d; px2: %d; py1: %d; py2: %d\n", i, input_box.px1, input_box.px2, input_box.py1, input_box.py2);

    chop_img = copy_img(cv::Range(input_box.px1, input_box.px2), cv::Range(input_box.py1, input_box.py2));

    // if (p_str == "R")
    // {
    //     string PicName_2 = "padface" + std::to_string(i) + "_" + std::to_string(i) + ".jpg";
    //     cv::imwrite(PicName_2, chop_img);
    // }

    chop_img.convertTo( chop_img, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
	cv::resize(chop_img, chop_img, cv::Size(width, height));

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                data_to[c*width*height + i*height + j] = (chop_img.ptr<float>(j)[i*3+c]);
            }
        }
    }
}

// compute the padding coordinates (pad the bounding boxes to square)
void Padding(vector<BoundingBox> &totalBoxes, int img_w, int img_h){
    for(int i = 0;i < totalBoxes.size(); i++)
    {
        totalBoxes[i].py2 = int((totalBoxes[i].y2 >= img_w) ? img_w : totalBoxes[i].y2);
        totalBoxes[i].px2 = int((totalBoxes[i].x2 >= img_h) ? img_h : totalBoxes[i].x2);
        totalBoxes[i].py1 = int((totalBoxes[i].y1 < 1) ? 1 : totalBoxes[i].y1);
        totalBoxes[i].px1 = int((totalBoxes[i].x1 < 1) ? 1 : totalBoxes[i].x1);
        //printf("===%d===px1: %d; px2: %d; py1: %d; py2: %d\n", i, totalBoxes[i].px1, totalBoxes[i].px2, totalBoxes[i].py1, totalBoxes[i].py2);
    }
}

int main()
{
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(1);
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_option, 0));
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    //PNet
    std::string m_pModel_dir = "./optimaizer_pnet.onnx";
    Ort::Session session_pnet(env, m_pModel_dir.c_str(), session_option);
    std::vector<const char*> m_PNetInputNodeNames;
    std::vector<const char*> m_PNetOutputNodeNames;
    GetOnnxModelInputInfo(session_pnet, m_PNetInputNodeNames, m_PNetOutputNodeNames);

    //图像处理部分，需要单独写一个类
    Mat img = imread("./Trump.jpeg");
    cv::Mat sample;
    sample = img.clone();//深拷贝

    //RGB->BGR
    vector<cv::Mat> sample_norm_channels;
    cv::split(sample, sample_norm_channels);
    cv::Mat tmp = sample_norm_channels[0];
    sample_norm_channels[0] = sample_norm_channels[2];
    sample_norm_channels[2] = tmp;

    cv::merge(sample_norm_channels, sample);
    imwrite("sample.jpg", sample);

    const int img_H = img.rows;
    const int img_W = img.cols;

    int minl  = cv::min(img_H, img_W);

    const int minsize = 20;
    double m = 12.0/minsize;
    minl = minl * m;
    vector<double> scales;
    int factor_count = 0;
    double factor = 0.709;
    while(minl >= 12.0)
    {
        scales.push_back(m * pow(factor , factor_count));
        minl *= factor;
        factor_count += 1;
    }

    int mi = 0;
    printf("scales.size: %d\n", (int)scales.size());
    printf("scales.size: ");
    for (auto it : scales)
    {
        printf(" %f ", it);
    }
    printf("\n");

    vector<BoundingBox> totalBoxes;
    for (auto scale : scales)
    {
        printf("-----start-----\n");
        printf("scales:%.17f\n", scale);
        int hs = ceil(img_H * scale);
        printf("hs: %d, ", hs);
        int ws = ceil(img_W * scale);
        printf("ws: %d\n", ws);

        //convert to float and normalize
        cv::Mat OutputPic, OutputPic_2;
        resize(sample, OutputPic, Size(ws, hs));
        //string PicName = "wytface" + std::to_string(mi) + ".jpg";
        //cv::imwrite(PicName, OutputPic);
        OutputPic.convertTo( OutputPic_2, CV_32FC3, 0.0078125, -127.5 * 0.0078125);

        size_t input_tensor_size = 1 * 3 * ws * hs;
        std::vector<float> input_image_(input_tensor_size);
        std::array<int64_t, 4> input_shape_{ 1, 3, ws, hs };

        float* output = input_image_.data();
        fill(input_image_.begin(), input_image_.end(), 0.f);
        //HWC->NCWH
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < ws; i++) {
                for (int j = 0; j < hs; j++) {
                    output[c*ws*hs + i*hs + j] = (OutputPic_2.ptr<float>(j)[i*3+c]);
                }
            }
        }

        // create input tensor object from data values
        Ort::Value input_tensor_pnet = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

        auto input_type_info = input_tensor_pnet.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_type_info_dims = input_type_info.GetShape();
        for (int i = 0; i < input_type_info_dims.size(); i++)
        {
           printf("input_type_info_dims shape [%d] =  %ld\n", i, input_type_info_dims[i]);
        }
        
        auto output_tensors_pnet = session_pnet.Run(Ort::RunOptions{nullptr}, m_PNetInputNodeNames.data(), &input_tensor_pnet, 1, m_PNetOutputNodeNames.data(), m_PNetOutputNodeNames.size());
        printf("output_tensors_pnet.size: [%d]\n", (int)output_tensors_pnet.size());

        //2-Y 框
        printf("---------------2-Y-------------------\n");
        printf("Output %s:\n", m_PNetOutputNodeNames[0]);
        Ort::TypeInfo type_info_0 = output_tensors_pnet[0].GetTypeInfo();
        auto tensor_info_0 = type_info_0.GetTensorTypeAndShapeInfo();
        size_t tensor_size_0 = tensor_info_0.GetElementCount();
        vector<int64_t>  output_node_dims_0 = tensor_info_0.GetShape();
        //printf("output_node_dims_0: %d \n", (int)output_node_dims_0.size());
        // printf("output_node_dims_0 shape: ");
        // for(int i=0; i < output_node_dims_0.size(); i++)
        // {
        //     printf("  %ld  ", output_node_dims_0[i]);
        // }
        // printf("\n");
        float *outarr0 = output_tensors_pnet[0].GetTensorMutableData<float>();
        printf("tensor_size_0: %d \n", (int)tensor_size_0);
        vector<float> out0{outarr0, outarr0 + tensor_size_0};
        printf("out0_size: %d \n", (int)out0.size());

        //P-Y softmaxOutput
        //printf("----------------P-Y------------------\n");
        //printf("Output %s:\n", m_PNetOutputNodeNames[1]);
        Ort::TypeInfo type_info_1 = output_tensors_pnet[1].GetTypeInfo();
        auto tensor_info_1 = type_info_1.GetTensorTypeAndShapeInfo();
        size_t tensor_size_1 = tensor_info_1.GetElementCount();
        vector<int64_t>  output_node_dims_1 = tensor_info_1.GetShape();
        //printf("output_node_dims_1: %d \n", (int)output_node_dims_1.size());
        //printf("output_node_dims_1 shape: ");
        // for(int i=0; i < output_node_dims_1.size(); i++)
        // {
        //     printf("  %ld  ", output_node_dims_1[i]);
        // }
        // printf("\n");
        float *outarr1 = output_tensors_pnet[1].GetTensorMutableData<float>();
        //printf("tensor_size_1: %d \n", (int)tensor_size_1);
        vector<float> out1{outarr1, outarr1 + tensor_size_1};
        printf("out0_size: %d \n", (int)out1.size());
        
        //1-Y nosoftmax
        //printf("----------------1-Y------------------\n");
        //printf("Output %s:\n", m_PNetOutputNodeNames[2]);
        Ort::TypeInfo type_info_2 = output_tensors_pnet[2].GetTypeInfo();
        auto tensor_info_2 = type_info_2.GetTensorTypeAndShapeInfo();
        size_t tensor_size_2 = tensor_info_2.GetElementCount();
        vector<int64_t>  output_node_dims_2 = tensor_info_2.GetShape();
        //printf("output_node_dims_2: %d \n", (int)output_node_dims_2.size());
        // printf("output_node_dims_2 shape: ");
        // for(int i=0; i < output_node_dims_2.size(); i++)
        // {
        //     printf("  %ld  ", output_node_dims_2[i]);
        // }
        // printf("\n");
        float *outarr2 = output_tensors_pnet[2].GetTensorMutableData<float>();
        printf("tensor_size_2: %d \n", (int)tensor_size_2);
        vector<float> out2{outarr2, outarr2 + tensor_size_2};

        softmax(out2);

        vector<BoundingBox> filterOutBoxes;
        vector<BoundingBox> nmsOutBoxes;
        generateBoundingBox(out0, output_node_dims_0, out2, output_node_dims_2, scale, 0.6, filterOutBoxes);
        printf("-------filterOutBoxes.size(): %d\n", (int)filterOutBoxes.size());
        nmsOutBoxes = nms(filterOutBoxes, 0.5, UNION);
        printf("-------nmsOutBoxes.size(): %d\n", (int)nmsOutBoxes.size());
        if(nmsOutBoxes.size() > 0)
        {
            totalBoxes.insert(totalBoxes.end(), nmsOutBoxes.begin(), nmsOutBoxes.end());
            printf("======1=====>totalBoxes.size(): %d\n", (int)totalBoxes.size());
        }

        mi++;
        printf("-----end-----\n");
    }

    //do global nms operator
    if (totalBoxes.size() > 0)
    {
        vector<BoundingBox> globalFilterBoxes;
        printf("======2=====>totalBoxes.size(): %d\n", (int)totalBoxes.size());
        globalFilterBoxes = nms(totalBoxes, 0.7, UNION);
        printf("globalFilterBoxes.size(): %ld\n", globalFilterBoxes.size());
        totalBoxes.clear();
        for(int i = 0; i < globalFilterBoxes.size(); i ++)
        {
            float regw = globalFilterBoxes[i].y2 - globalFilterBoxes[i].y1 ;
            float regh = globalFilterBoxes[i].x2 - globalFilterBoxes[i].x1;
            BoundingBox box;
            float x1 = globalFilterBoxes[i].x1 + globalFilterBoxes[i].dy1 * regw;
            float y1 = globalFilterBoxes[i].y1 + globalFilterBoxes[i].dx1 * regh;
            float x2 = globalFilterBoxes[i].x2 + globalFilterBoxes[i].dy2 * regw;
            float y2 = globalFilterBoxes[i].y2 + globalFilterBoxes[i].dx2 * regh;
            float score = globalFilterBoxes[i].score;

            float h = y2 - y1;
            float w = x2 - x1;
            float l = std::max(h, w);
            x1 += (w - l) * 0.5;
            y1 += (h - l) * 0.5;
            x2 = x1 + l;
            y2 = y1 + l;
            //box.x1 = y1, box.x2 = y2, box.y1 = x1, box.y2 = x2;
            box.x1 = x1, box.x2 = x2, box.y1 = y1, box.y2 = y2;
            //printf("===%d===box.x1: %f; box.x2: %f; box.y1: %f; box.y2: %f\n", i, box.x1, box.x2, box.y1, box.y2);
            totalBoxes.push_back(box);
        }

        printf("totalBoxes.size(): %d\n", (int)totalBoxes.size());
    }
    
    // {
    //     cv::Mat m_tmp = img.t();
    //     for(int k = 0; k < totalBoxes.size(); k++)
    //     {
    //         cv::rectangle(m_tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 0), 2);
    //     }
    //     imwrite("wytFace_t.jpg", m_tmp.t());
    // }

    //R-Net
    if(totalBoxes.size() > 0)
    {
        int batch = totalBoxes.size();
        int channel = 3;
        int height_r = 24;
        int width_r = 24;

        float rnet_threshold=0.7;

        std::string m_rModel_dir = "./optimaizer_rnet.onnx";
        Ort::Session session_rnet(env, m_rModel_dir.c_str(), session_option);
        std::vector<const char*> m_RNetInputNodeNames;
        std::vector<const char*> m_RNetOutputNodeNames;
        GetOnnxModelInputInfo(session_rnet, m_RNetInputNodeNames, m_RNetOutputNodeNames);
        std::array<int64_t, 4> input_shape_{ 1, channel, height_r, width_r };

        vector<int64_t>  output_node_dims_0;
        vector<int64_t>  output_node_dims_1;
        vector<float> out0;
        vector<float> out1;

        Padding(totalBoxes, img_W, img_H);
        for (int i = 0; i < batch; i++)
        {
            printf("==============batch===%d=========>\n", i);
            size_t input_tensor_size = 1 * channel * height_r * width_r;
            std::vector<float> input_image_(input_tensor_size);

            float *input_data = input_image_.data();
            fill(input_image_.begin(), input_image_.end(), 0.f);

            copy_one_patch(sample, totalBoxes[i], input_data, cv::Size(24, 24), i,  "R");
            // printf("input_data: \n");
            // for (int j = 0; j < input_tensor_size; j ++)
            // {
            //     printf(" %f ", input_data[j]);
            // }
            // printf("\n");
            // //return 0;

            // create input tensor object from data values
            Ort::Value input_tensor_rnet = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

            auto output_tensors_rnet = session_rnet.Run(Ort::RunOptions{nullptr}, m_RNetInputNodeNames.data(), &input_tensor_rnet, 1, m_RNetOutputNodeNames.data(), m_RNetOutputNodeNames.size());
            // printf("output_tensors_rnet.size: [%d]\n", (int)output_tensors_rnet.size());

            //conv5-2_Gemm_Y
            //printf("---------------conv5-2_Gemm_Y-------------------\n");
            //printf("Output %s:\n", m_RNetOutputNodeNames[0]);
            Ort::TypeInfo type_info_0 = output_tensors_rnet[0].GetTypeInfo();
            auto tensor_info_0 = type_info_0.GetTensorTypeAndShapeInfo();
            size_t tensor_size_0 = tensor_info_0.GetElementCount();
            vector<int64_t>  m_vecOut0 = tensor_info_0.GetShape();
            for (auto it : m_vecOut0)
                output_node_dims_0.push_back(it);

            float *outarr0 = output_tensors_rnet[0].GetTensorMutableData<float>();
            printf("tensor_size_0: %d \n", (int)tensor_size_0);
            //printf("pos: \n");
            for (int j = 0; j < tensor_size_0; j++)
            {
                out0.push_back(outarr0[j]);
                //printf(" %f ", outarr0[j]);
            }
            //printf("\n");

            //prob1_Y
            //printf("----------------prob1_Y------------------\n");
            //printf("Output %s:\n", m_RNetOutputNodeNames[1]);
            Ort::TypeInfo type_info_1 = output_tensors_rnet[1].GetTypeInfo();
            auto tensor_info_1 = type_info_1.GetTensorTypeAndShapeInfo();
            size_t tensor_size_1 = tensor_info_1.GetElementCount();
            vector<int64_t>  m_vecOut1 = tensor_info_1.GetShape();
            for (auto it : m_vecOut1)
            {
                output_node_dims_1.push_back(it);
            }

            float *outarr1 = output_tensors_rnet[1].GetTensorMutableData<float>();
            //printf("tensor_size_1: %d \n", (int)tensor_size_1);
            //printf("score: \n");
            for (int j = 0; j < tensor_size_1; j++)
            {
                out1.push_back(outarr1[j]);
                //printf(" %f ", outarr1[j]);
            }
            //printf("\n");
        }

        vector<BoundingBox> filterOutBoxes;
        filteroutBoundingBox(totalBoxes, out0, output_node_dims_0, out1, output_node_dims_1, vector<float>(), vector<int64_t>(), 0.7, filterOutBoxes);
        printf("filterOutBoxes.size = %zu \n", filterOutBoxes.size());
        totalBoxes.clear();
        totalBoxes = nms(filterOutBoxes, 0.7, UNION);
        printf("totalBoxes.size = %zu \n", totalBoxes.size());

        cv::Mat m_tmp = img.t();
        for(int k = 0; k < totalBoxes.size(); k++)
        {
            cv::rectangle(m_tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 0), 2);
        }
        imwrite("wytFace_Rnet_Out.jpg", m_tmp.t());
    }

    //O-Net
    if (totalBoxes.size() > 0)
    {
        int batch = totalBoxes.size();
        int channel = 3;
        int height_r = 48;
        int width_r = 48;

        float rnet_threshold=0.7;

        std::string m_oModel_dir = "./optimaizer_onet.onnx";
        Ort::Session session_onet(env, m_oModel_dir.c_str(), session_option);
        std::vector<const char*> m_ONetInputNodeNames;
        std::vector<const char*> m_ONetOutputNodeNames;
        GetOnnxModelInputInfo(session_onet, m_ONetInputNodeNames, m_ONetOutputNodeNames);
        std::array<int64_t, 4> input_shape_{ 1, channel, height_r, width_r };

        vector<int64_t>  output_node_dims_0;
        vector<int64_t>  output_node_dims_1;
        vector<int64_t>  output_node_dims_2;
        vector<float> out0;
        vector<float> out1;
        vector<float> out2;

        Padding(totalBoxes, img_W, img_H);
        for (int i = 0; i < batch; i++)
        {
            printf("==============batch===%d=========>\n", i);
            size_t input_tensor_size = 1 * channel * height_r * width_r;
            std::vector<float> input_image_(input_tensor_size);

            float *input_data = input_image_.data();
            fill(input_image_.begin(), input_image_.end(), 0.f);

            copy_one_patch(sample, totalBoxes[i], input_data, cv::Size(48, 48), i, "O");
            // printf("input_data: \n");
            // for (int j = 0; j < input_tensor_size; j ++)
            // {
            //     printf(" %f ", input_data[j]);
            // }
            // printf("\n");
            // return 0;

            // create input tensor object from data values
            Ort::Value input_tensor_onet = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

            auto output_tensors_onet = session_onet.Run(Ort::RunOptions{nullptr}, m_ONetInputNodeNames.data(), &input_tensor_onet, m_ONetInputNodeNames.size(), m_ONetOutputNodeNames.data(), m_ONetOutputNodeNames.size());

            //conv6-2_Gemm_Y  boxes
            printf("---------------conv6-2_Gemm_Y-------------------\n");
            printf("Output %s:\n", m_ONetOutputNodeNames[0]);
            Ort::TypeInfo type_info_0 = output_tensors_onet[0].GetTypeInfo();
            auto tensor_info_0 = type_info_0.GetTensorTypeAndShapeInfo();
            size_t tensor_size_0 = tensor_info_0.GetElementCount();
            vector<int64_t>  m_vecOut0 = tensor_info_0.GetShape();
            for (auto it : m_vecOut0)
            {
                output_node_dims_0.push_back(it);
            }

            float *outarr0 = output_tensors_onet[0].GetTensorMutableData<float>();
            printf("tensor_size_0: %d \n", (int)tensor_size_0);
            printf("conv6-2_Gemm_Y boxes: \n");
            for (int j = 0; j < tensor_size_0; j++)
            {
                out0.push_back(outarr0[j]);
                printf(" %f ", outarr0[j]);
            }
            printf("\n");

            // conv6-3_Gemm_Y landmark
            printf("----------------conv6-3_Gemm_Y------------------\n");
            printf("Output %s:\n", m_ONetOutputNodeNames[1]);
            Ort::TypeInfo type_info_1 = output_tensors_onet[1].GetTypeInfo();
            auto tensor_info_1 = type_info_1.GetTensorTypeAndShapeInfo();
            size_t tensor_size_1 = tensor_info_1.GetElementCount();
            vector<int64_t>  m_vecOut1 = tensor_info_1.GetShape();
            for (auto it : m_vecOut1)
            {
                output_node_dims_1.push_back(it);
            }

            float *outarr1 = output_tensors_onet[1].GetTensorMutableData<float>();
            printf("tensor_size_1: %d \n", (int)tensor_size_1);
            printf("conv6-3_Gemm_Y landmark: \n");
            for (int j = 0; j < tensor_size_1; j++)
            {
                out1.push_back(outarr1[j]);
                printf(" %f ", outarr1[j]);
            }
            printf("\n");

            // prob1_Y prob
            printf("----------------prob1_Y------------------\n");
            printf("Output %s:\n", m_ONetOutputNodeNames[2]);
            Ort::TypeInfo type_info_2 = output_tensors_onet[2].GetTypeInfo();
            auto tensor_info_2 = type_info_2.GetTensorTypeAndShapeInfo();
            size_t tensor_size_2 = tensor_info_2.GetElementCount();
            vector<int64_t>  m_vecOut2 = tensor_info_2.GetShape();
            for (auto it : m_vecOut2)
            {
                output_node_dims_2.push_back(it);
            }
            
            float *outarr2 = output_tensors_onet[2].GetTensorMutableData<float>();
            printf("tensor_size_2: %d \n", (int)tensor_size_2);
            printf("prob1_Y prob: \n");
            for (int j = 0; j < tensor_size_2; j++)
            {
                out2.push_back(outarr2[j]);
                printf(" %f ", outarr2[j]);
            }
            printf("\n==================end=================\n");
        }

        vector<BoundingBox> filterOutBoxes;
        filteroutBoundingBox(totalBoxes, out0, output_node_dims_0, out2, output_node_dims_2, out1, output_node_dims_1, 0.7, filterOutBoxes);
        printf("filterOutBoxes.size = %zu \n", filterOutBoxes.size());
        totalBoxes.clear();
        totalBoxes = nms(filterOutBoxes, 0.7, MIN);
        printf("totalBoxes.size = %zu \n", totalBoxes.size());

        cv::Mat m_tmp = img.t();
        for(int k = 0; k < totalBoxes.size(); k++)
        {
            cv::rectangle(m_tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 0), 2);
        }
        imwrite("wytFace_Onet_Out.jpg", m_tmp.t());
    }

    for(int i = 0; i < totalBoxes.size(); i++)
    {
        std::swap(totalBoxes[i].x1, totalBoxes[i].y1);
        std::swap(totalBoxes[i].x2, totalBoxes[i].y2);
        for(int k = 0; k < 5; k++)
        {
            std::swap(totalBoxes[i].points_x[k], totalBoxes[i].points_y[k]);
        }
    }

    cv::Mat m_tmp = img.clone();
    for(int k = 0; k < totalBoxes.size(); k++)
    {
        cv::rectangle(m_tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 255), 2);
        for(int i = 0; i < 5; i ++)
            cv::circle(m_tmp, cv::Point(totalBoxes[k].points_x[i], totalBoxes[k].points_y[i]), 2, cv::Scalar(0, 255, 255), 2);
    }
    imwrite("wytFace_Out.jpg", m_tmp);

    return 0;
}

