#include "facedetector.h"

FaceDetector::FaceDetector(std::vector<string> model_dir)
{
    m_model_dir.assign(model_dir.begin(), model_dir.end());
    cout << "m_model_dir :" << m_model_dir.size() << endl;

    num_channels_ = 3;
    //set img_mean
    img_mean = 127.5;
    //set img_var
    img_var  = 0.0078125;
}

FaceDetector::~FaceDetector()
{
    //Release();
    session_PNet.reset();
    session_RNet.reset();
    session_ONet.reset();
    m_OrtEnv.reset();
}

void FaceDetector::Init()
{
    cout << "Init()......" << endl;
    m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

    session_option.SetIntraOpNumThreads(1);
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));
    
    if (YKX_SUCCESS != GetOnnxModelInfo(m_model_dir))
    {
        printf("ModelInit failed!!!\n");
        return;
    }
}

void FaceDetector::copy_one_patch(const cv::Mat& img, BoundingBox& input_box, float *data_to, cv::Size target_size, int idx, const char *p_str)
{
    cv::Mat copy_img = img.clone();
    cv::Mat chop_img;

    int height = target_size.height;
    int width = target_size.width;
    float src_height = abs(input_box.px1 - input_box.px2);
    float src_width = abs(input_box.py1 - input_box.py2);
    chop_img = copy_img(cv::Range(input_box.px1, input_box.px2), cv::Range(input_box.py1, input_box.py2));

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

void FaceDetector::generateBoundingBox(const vector<float>& boxRegs, const vector<int64_t>& box_shape,
                             const vector<float>& cls, const vector<int64_t>& cls_shape,
                             float scale, float threshold, vector<BoundingBox>& filterOutBoxes
                            )
{
    //clear output element
    filterOutBoxes.clear();
    int stride = 2;
    int cellsize = 12;
    int w = box_shape[3];
    int h = box_shape[2];

    for(int y = 0; y < h; y ++)
    {
        for(int x = 0; x < w; x ++)
        {
            float score = cls[0 * 2 * w * h + 1 * w * h + w * y + x];
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
}

void FaceDetector::filteroutBoundingBox(const vector< FaceDetector::BoundingBox >& boxes, 
                                        const vector< float >& boxRegs, const vector< int64_t >& box_shape, 
                                        const vector< float >& cls, const vector< int64_t >& cls_shape, 
                                        const vector< float >& points, const vector< int64_t >& points_shape,
                                        float threshold, vector< FaceDetector::BoundingBox >& filterOutBoxes)
{
    filterOutBoxes.clear();

    for(int i = 0; i < boxes.size(); i ++)
    {
        float score = cls[i * 2 + 1];
        if ( score > threshold )
        {
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
            box.dy1 = boxRegs[i * 4 + 1];
            box.dx2 = boxRegs[i * 4 + 2];
            box.dy2 = boxRegs[i * 4 + 3];

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

float FaceDetector::iou(BoundingBox box1, BoundingBox box2, NMS_TYPE type) 
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

vector<FaceDetector::BoundingBox> FaceDetector::nms(std::vector<BoundingBox>& vec_boxs, float threshold, NMS_TYPE type) 
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

void FaceDetector::GetOnnxModelInputInfo(Ort::Session &session_net, 
                                            std::vector<const char*> &input_node_names, 
                                            vector<vector<int64_t> > &input_node_dims, 
                                            std::vector<const char*> &output_node_names, 
                                            vector<vector<int64_t> > &output_node_dims)
{
    size_t num_input_nodes = session_net.GetInputCount();
    input_node_names.resize(num_input_nodes);

    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "Number of inputs :" << num_input_nodes << std::endl;
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
        std::vector<int64_t> inputNodeDims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, inputNodeDims.size());
        for (int j = 0; j < inputNodeDims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, inputNodeDims[j]);
        }
        input_node_dims.push_back(inputNodeDims);
    }

    size_t num_output_nodes = session_net.GetOutputCount();
    output_node_names.resize(num_output_nodes);
    //std::vector<int64_t> output_node_dims;
    //char* output_name = nullptr;

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
        std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
        printf("output %d : num_dims=%zu\n", i, outputNodeDims.size());
        for (int j = 0; j < outputNodeDims.size(); j++)
        {
            printf("output %d : dim %d=%jd\n", i, j, outputNodeDims[j]);
        }
        output_node_dims.push_back(outputNodeDims);
    }

}

int64_t FaceDetector::GetOnnxModelInfo(std::vector<string> model_dir)
{
    if (model_dir.empty())
    {
        cout << "model_dir empty, please check it!!!" << endl;
        return YKX_PARAM_ERROR;
    }

    //GetModel dir
    m_pModel_dir = model_dir[0];
    m_rModel_dir = model_dir[1];
    m_oModel_dir = model_dir[2];

    /* load three networks */
    //p
    session_PNet = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, m_pModel_dir.c_str(), session_option));
    //R
    session_RNet = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, m_rModel_dir.c_str(), session_option));
    //O
    session_ONet = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, m_oModel_dir.c_str(), session_option));

    GetOnnxModelInputInfo(*session_PNet, m_PNetInputNodeNames, m_PNetInputNodesDims, m_PNetOutputNodeNames, m_PNetOutputNodesDims);
    GetOnnxModelInputInfo(*session_RNet, m_RNetInputNodeNames, m_RNetInputNodesDims, m_RNetOutputNodeNames, m_RNetOutputNodesDims);
    GetOnnxModelInputInfo(*session_ONet, m_ONetInputNodeNames, m_ONetInputNodesDims, m_ONetOutputNodeNames, m_ONetOutputNodesDims);

    return YKX_SUCCESS;
}

// compute the padding coordinates (pad the bounding boxes to square)
void FaceDetector::Padding(vector<BoundingBox> &totalBoxes, int img_w, int img_h)
{
    for(int i = 0;i < totalBoxes.size(); i++)
    {
        totalBoxes[i].py2 = int((totalBoxes[i].y2 >= img_w) ? img_w : totalBoxes[i].y2);
        totalBoxes[i].px2 = int((totalBoxes[i].x2 >= img_h) ? img_h : totalBoxes[i].x2);
        totalBoxes[i].py1 = int((totalBoxes[i].y1 < 1) ? 1 : totalBoxes[i].y1);
        totalBoxes[i].px1 = int((totalBoxes[i].x1 < 1) ? 1 : totalBoxes[i].x1);
        //printf("===%d===px1: %d; px2: %d; py1: %d; py2: %d\n", i, totalBoxes[i].px1, totalBoxes[i].px2, totalBoxes[i].py1, totalBoxes[i].py2);
    }
}

//#define IMAGE_DEBUG
vector< FaceDetector::BoundingBox > FaceDetector::Detect(const cv::Mat& img, const COLOR_ORDER color_order,  const IMAGE_DIRECTION orient,\
                                                            int min_size, float P_thres, float R_thres, float O_thres, bool is_fast_resize,\
                                                             float scale_factor)
{
    /*change image format*/
    cv::Mat sample;
    sample = img.clone();
    cv::Mat sample_normalized;

    //split the input image RGB->BGR
    vector<cv::Mat> sample_norm_channels;
    cv::split(sample, sample_norm_channels);
    if(color_order == BGR || color_order == BGRA)
    {
        cv::Mat tmp = sample_norm_channels[0];
        sample_norm_channels[0] = sample_norm_channels[2];
        sample_norm_channels[2] = tmp;
    }
    cv::merge(sample_norm_channels, sample);

    vector<float> points;
    const int img_H = img.rows;
    const int img_W = img.cols;
    int minl  = cv::min(img_H, img_W);
    
    float m = 12.0 / min_size;
    minl *= m;
    vector<float> all_scales;
    int factor_count = 0;
    while(minl >= 12.0)
    {
        all_scales.push_back(m * pow(scale_factor , factor_count));
        minl *= scale_factor;
        factor_count += 1;
    }
    /*stage 1: P_Net forward can get rectangle and regression */
    vector<BoundingBox> totalBoxes;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    for (auto cur_scale : all_scales)
    {

        int hs = cvCeil(img_H * cur_scale);
        int ws = cvCeil(img_W * cur_scale);
        cv::Mat OutputPic, OutputPic_2;
        resize(sample, OutputPic, Size(ws, hs));
        //convert to float and normalize
        OutputPic.convertTo( OutputPic_2, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
        //对输入的形状进行变化
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
        
        auto output_tensors_pnet = session_PNet->Run(Ort::RunOptions{nullptr}, m_PNetInputNodeNames.data(), &input_tensor_pnet, 1, m_PNetOutputNodeNames.data(), m_PNetOutputNodeNames.size());

        //2-Y 框
        Ort::TypeInfo type_info_0 = output_tensors_pnet[0].GetTypeInfo();
        auto tensor_info_0 = type_info_0.GetTensorTypeAndShapeInfo();
        size_t tensor_size_0 = tensor_info_0.GetElementCount();
        vector<int64_t>  output_node_dims_0 = tensor_info_0.GetShape();

        float *outarr0 = output_tensors_pnet[0].GetTensorMutableData<float>();
        vector<float> out0{outarr0, outarr0 + tensor_size_0};

        //P-Y softmaxOutput
        Ort::TypeInfo type_info_1 = output_tensors_pnet[1].GetTypeInfo();
        auto tensor_info_1 = type_info_1.GetTensorTypeAndShapeInfo();
        size_t tensor_size_1 = tensor_info_1.GetElementCount();
        vector<int64_t>  output_node_dims_1 = tensor_info_1.GetShape();

        float *outarr1 = output_tensors_pnet[1].GetTensorMutableData<float>();
        vector<float> out1{outarr1, outarr1 + tensor_size_1};
        
        //1-Y nosoftmax
        Ort::TypeInfo type_info_2 = output_tensors_pnet[2].GetTypeInfo();
        auto tensor_info_2 = type_info_2.GetTensorTypeAndShapeInfo();
        size_t tensor_size_2 = tensor_info_2.GetElementCount();
        vector<int64_t>  output_node_dims_2 = tensor_info_2.GetShape();
        float *outarr2 = output_tensors_pnet[2].GetTensorMutableData<float>();
        vector<float> out2{outarr2, outarr2 + tensor_size_2};

        softmax(out2);

        vector<BoundingBox> filterOutBoxes;
        vector<BoundingBox> nmsOutBoxes;
        generateBoundingBox(out0, output_node_dims_0, out2, output_node_dims_2, cur_scale, P_thres, filterOutBoxes);
        nmsOutBoxes = nms(filterOutBoxes, 0.5, UNION);
        if(nmsOutBoxes.size() > 0)
        {
            totalBoxes.insert(totalBoxes.end(), nmsOutBoxes.begin(), nmsOutBoxes.end());
        }

    }

    //do global nms operator
    if (totalBoxes.size() > 0)
    {
        vector<BoundingBox> globalFilterBoxes;
        //cout<<totalBoxes.size()<<endl;
        globalFilterBoxes = nms(totalBoxes, 0.7, UNION);
        //cout<<globalFilterBoxes.size()<<endl;
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
            box.x1 = x1, box.x2 = x2, box.y1 = y1, box.y2 = y2;
            totalBoxes.push_back(box);
        }
    }

    // cv::Mat m_tmp = img.t();
    // for(int k = 0; k < totalBoxes.size(); k++)
    // {
    //     cv::rectangle(m_tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 0), 2);
    // }
    // imwrite("wytFace_Pnet_Out.jpg", m_tmp.t());

    //the second stage: R-Net
    if(totalBoxes.size() > 0)
    {
        int batch = totalBoxes.size();
        int channel = 3;
        int height_r = 24;
        int width_r = 24;

        std::array<int64_t, 4> input_shape_{ 1, channel, height_r, width_r };

        vector<int64_t>  output_node_dims_0;
        vector<int64_t>  output_node_dims_1;
        vector<float> out0;
        vector<float> out1;

        Padding(totalBoxes, img_W, img_H);
        for (int i = 0; i < batch; i++)
        {
            size_t input_tensor_size = 1 * channel * height_r * width_r;
            std::vector<float> input_image_(input_tensor_size);

            float *input_data = input_image_.data();
            fill(input_image_.begin(), input_image_.end(), 0.f);

            copy_one_patch(sample, totalBoxes[i], input_data, cv::Size(24, 24), i,  "R");

            // create input tensor object from data values
            Ort::Value input_tensor_rnet = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

            auto output_tensors_rnet = session_RNet->Run(Ort::RunOptions{nullptr}, m_RNetInputNodeNames.data(), &input_tensor_rnet, 1, m_RNetOutputNodeNames.data(), m_RNetOutputNodeNames.size());

            //conv5-2_Gemm_Y
            Ort::TypeInfo type_info_0 = output_tensors_rnet[0].GetTypeInfo();
            auto tensor_info_0 = type_info_0.GetTensorTypeAndShapeInfo();
            size_t tensor_size_0 = tensor_info_0.GetElementCount();
            vector<int64_t>  m_vecOut0 = tensor_info_0.GetShape();
            for (auto it : m_vecOut0)
                output_node_dims_0.push_back(it);

            float *outarr0 = output_tensors_rnet[0].GetTensorMutableData<float>();
            for (int j = 0; j < tensor_size_0; j++)
            {
                out0.push_back(outarr0[j]);
            }

            //prob1_Y
            Ort::TypeInfo type_info_1 = output_tensors_rnet[1].GetTypeInfo();
            auto tensor_info_1 = type_info_1.GetTensorTypeAndShapeInfo();
            size_t tensor_size_1 = tensor_info_1.GetElementCount();
            vector<int64_t>  m_vecOut1 = tensor_info_1.GetShape();
            for (auto it : m_vecOut1)
            {
                output_node_dims_1.push_back(it);
            }

            float *outarr1 = output_tensors_rnet[1].GetTensorMutableData<float>();
            for (int j = 0; j < tensor_size_1; j++)
            {
                out1.push_back(outarr1[j]);
            }
        }
        vector<BoundingBox> filterOutBoxes;
        filteroutBoundingBox(totalBoxes, out0, output_node_dims_0, out1, output_node_dims_1, vector<float>(), vector<int64_t>(), R_thres, filterOutBoxes);
        totalBoxes.clear();
        totalBoxes = nms(filterOutBoxes, 0.7, UNION);

        // cv::Mat m_tmp = img.t();
        // for(int k = 0; k < totalBoxes.size(); k++)
        // {
        //     cv::rectangle(m_tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 0), 2);
        // }
        // imwrite("wytFace_Rnet_Out.jpg", m_tmp.t());

    }

    // do third stage: O-Net
    if(totalBoxes.size() > 0)
    {
        int batch = totalBoxes.size();
        int channel = 3;
        int height_o = 48;
        int width_o = 48;

        std::array<int64_t, 4> input_shape_{ 1, channel, height_o, width_o };

        vector<int64_t>  output_node_dims_0;
        vector<int64_t>  output_node_dims_1;
        vector<int64_t>  output_node_dims_2;
        vector<float> out0;
        vector<float> out1;
        vector<float> out2;

        Padding(totalBoxes, img_W, img_H);
        for (int i = 0; i < batch; i++)
        {
            size_t input_tensor_size = 1 * channel * height_o * width_o;
            std::vector<float> input_image_(input_tensor_size);

            float *input_data = input_image_.data();
            fill(input_image_.begin(), input_image_.end(), 0.f);

            copy_one_patch(sample, totalBoxes[i], input_data, cv::Size(height_o, width_o), i, "O");

            // create input tensor object from data values
            Ort::Value input_tensor_onet = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

            auto output_tensors_onet = session_ONet->Run(Ort::RunOptions{nullptr}, m_ONetInputNodeNames.data(), &input_tensor_onet, m_ONetInputNodeNames.size(), m_ONetOutputNodeNames.data(), m_ONetOutputNodeNames.size());

            //conv6-2_Gemm_Y  boxes
            Ort::TypeInfo type_info_0 = output_tensors_onet[0].GetTypeInfo();
            auto tensor_info_0 = type_info_0.GetTensorTypeAndShapeInfo();
            size_t tensor_size_0 = tensor_info_0.GetElementCount();
            vector<int64_t>  m_vecOut0 = tensor_info_0.GetShape();
            for (auto it : m_vecOut0)
            {
                output_node_dims_0.push_back(it);
            }

            float *outarr0 = output_tensors_onet[0].GetTensorMutableData<float>();
            for (int j = 0; j < tensor_size_0; j++)
            {
                out0.push_back(outarr0[j]);
            }

            // conv6-3_Gemm_Y landmark
            Ort::TypeInfo type_info_1 = output_tensors_onet[1].GetTypeInfo();
            auto tensor_info_1 = type_info_1.GetTensorTypeAndShapeInfo();
            size_t tensor_size_1 = tensor_info_1.GetElementCount();
            vector<int64_t>  m_vecOut1 = tensor_info_1.GetShape();
            for (auto it : m_vecOut1)
            {
                output_node_dims_1.push_back(it);
            }

            float *outarr1 = output_tensors_onet[1].GetTensorMutableData<float>();
            for (int j = 0; j < tensor_size_1; j++)
            {
                out1.push_back(outarr1[j]);
            }

            // prob1_Y prob
            Ort::TypeInfo type_info_2 = output_tensors_onet[2].GetTypeInfo();
            auto tensor_info_2 = type_info_2.GetTensorTypeAndShapeInfo();
            size_t tensor_size_2 = tensor_info_2.GetElementCount();
            vector<int64_t>  m_vecOut2 = tensor_info_2.GetShape();
            for (auto it : m_vecOut2)
            {
                output_node_dims_2.push_back(it);
            }
            
            float *outarr2 = output_tensors_onet[2].GetTensorMutableData<float>();
            for (int j = 0; j < tensor_size_2; j++)
            {
                out2.push_back(outarr2[j]);
            }
        }

        vector<BoundingBox> filterOutBoxes;
        filteroutBoundingBox(totalBoxes, out0, output_node_dims_0, out2, output_node_dims_2, out1, output_node_dims_1, O_thres, filterOutBoxes);
        totalBoxes.clear();
        totalBoxes = nms(filterOutBoxes, 0.7, MIN);

        // cv::Mat m_tmp = img.t();
        // for(int k = 0; k < totalBoxes.size(); k++)
        // {
        //     cv::rectangle(m_tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 0), 2);
        // }
        // imwrite("wytFace_Onet_Out.jpg", m_tmp.t());

    }

    for(int i = 0; i < totalBoxes.size(); i++)
    {
        if(orient == ORIENT_UP){
            std::swap(totalBoxes[i].x1, totalBoxes[i].y1);
            std::swap(totalBoxes[i].x2, totalBoxes[i].y2);
            for(int k = 0; k < 5; k++)
            {
                std::swap(totalBoxes[i].points_x[k], totalBoxes[i].points_y[k]);
            }
        }
        else if(orient == ORIENT_DOWN){
            totalBoxes[i].x1 = img_W - totalBoxes[i].x1;
            totalBoxes[i].y1 = img_H - totalBoxes[i].y1;
            totalBoxes[i].x2 = img_W - totalBoxes[i].x2;
            totalBoxes[i].y2 = img_H - totalBoxes[i].y2;
            std::swap(totalBoxes[i].x1, totalBoxes[i].y1);
            std::swap(totalBoxes[i].x2, totalBoxes[i].y2);
            for(int k = 0; k < 5; k++)
            {
                totalBoxes[i].points_x[k] = img_W - totalBoxes[i].points_x[k];
                totalBoxes[i].points_y[k] = img_H - totalBoxes[i].points_y[k];
                std::swap(totalBoxes[i].points_x[k], totalBoxes[i].points_y[k]);
            }
        }
        else if(orient == ORIENT_LEFT){
            totalBoxes[i].x1 = img_W - totalBoxes[i].x1;
            totalBoxes[i].y1 = img_H - totalBoxes[i].y1;
            totalBoxes[i].x2 = img_W - totalBoxes[i].x2;
            totalBoxes[i].y2 = img_H - totalBoxes[i].y2;
            for(int k = 0; k < 5; k++)
            {
                totalBoxes[i].points_x[k] = img_W - totalBoxes[i].points_x[k];
                totalBoxes[i].points_y[k] = img_H - totalBoxes[i].points_y[k];
            }
        }
    }
    return totalBoxes;
}