#include <facedetector.h>

Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };

FaceDetector::FaceDetector(std::vector<string> model_dir, const MODEL_VERSION model_version)
{
    session_option.SetIntraOpNumThreads(1);
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    if (YKX_SUCCESS != GetOnnxModelInfo(model_dir))
    {
        printf("ModelInit failed!!!\n");
        return;
    }

    num_channels_ = 3;
    //set img_mean
    img_mean = 127.5;
    //set img_var
    img_var  = 0.0078125;
}

FaceDetector::~FaceDetector()
{
    //Release();
}

void FaceDetector::Release()
{
    if (!m_PNetInputNodeNames.empty())
    {
        for (auto it : m_PNetInputNodeNames)
        {
            free(it);
            it = nullptr;
        }
    }

    if (!m_PNetOutputNodeNames.empty())
    {
        for (auto it : m_PNetOutputNodeNames)
        {
            free(it);
            it = nullptr;
        }
    }

    if (!m_RNetInputNodeNames.empty())
    {
        for (auto it : m_RNetInputNodeNames)
        {
            free(it);
            it = nullptr;
        }
    }

    if (!m_RNetOutputNodeNames.empty())
    {
        for (auto it : m_RNetOutputNodeNames)
        {
            free(it);
            it = nullptr;
        }
    }

    if (!m_ONetInputNodeNames.empty())
    {
        for (auto it : m_ONetInputNodeNames)
        {
            free(it);
            it = nullptr;
        }
    }

    if (!m_ONetOutputNodeNames.empty())
    {
        for (auto it : m_ONetOutputNodeNames)
        {
            free(it);
            it = nullptr;
        }
    }
}

void FaceDetector::wrapInputLayer(boost::shared_ptr< Net<float> > net, vector< cv::Mat >* input_channels)
{
    Blob<float>* input_layer = net->input_blobs()[0];
    
    int width = input_layer->width();
    int height = input_layer->height();
    
    float* input_data = input_layer->mutable_cpu_data();
    for(int j = 0; j < input_layer->num(); j++)
    {
        for(int i = 0; i < input_layer->channels(); i ++)
        {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
    }
}

void FaceDetector::pyrDown(const vector<cv::Mat>& img_channels, float scale, vector< cv::Mat >* input_channels)
{
    assert(img_channels.size() == input_channels->size());
    int hs = (*input_channels)[0].rows;
    int ws = (*input_channels)[0].cols;
    cv::Mat img_resized;
    for(int i = 0; i < img_channels.size(); i ++)
    {
        cv::resize(img_channels[i], (*input_channels)[i], cv::Size(ws, hs));
    }
}

void FaceDetector::buildInputChannels(const vector< cv::Mat >& img_channels, const std::vector<BoundingBox>& boxes,
                                      const cv::Size& target_size, vector< cv::Mat >* input_channels)
{
    //assert(img_channels.size() * boxes.size() == input_channels->size() );
    cv::Rect img_rect(0, 0, img_channels[0].cols, img_channels[0].rows);
    for(int n = 0; n < boxes.size(); n++)
    {
        cv::Rect rect;
        rect.x = boxes[n].x1;
        rect.y = boxes[n].y1;
        rect.width = boxes[n].x2 - boxes[n].x1 + 1;
        rect.height = boxes[n].y2 - boxes[n].y1 + 1;
        cv::Rect cuted_rect = rect & img_rect;
        cv::Rect inner_rect(cuted_rect.x - rect.x, cuted_rect.y - rect.y, cuted_rect.width, cuted_rect.height);
        for(int c = 0; c < img_channels.size(); c++)
        {
            cv::Mat tmp(rect.height, rect.width, CV_32FC1, cv::Scalar(0.0));
            img_channels[c](cuted_rect).copyTo(tmp(inner_rect));
            cv::resize(tmp, (*input_channels)[n * img_channels.size() + c], target_size);
        }
    }
}

void FaceDetector::generateBoundingBox(const vector<float>& boxRegs, const vector<int>& box_shape,
                             const vector<float>& cls, const vector<int>& cls_shape,
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
    int h = box_shape[2];
    //int n = box_shape[0];
    for(int y = 0; y < h; y ++)
    {
        for(int x = 0; x < w; x ++)
        {
            float score =     cls[0 * 2 * w * h + 1 * w * h + w * y + x];
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
                                        const vector< float >& boxRegs, const vector< int >& box_shape, 
                                        const vector< float >& cls, const vector< int >& cls_shape, 
                                        const vector< float >& points, const vector< int >& points_shape,
                                        float threshold, vector< FaceDetector::BoundingBox >& filterOutBoxes)
{
    filterOutBoxes.clear();
    assert(box_shape.size() == cls_shape.size());
    assert(box_shape[0] == boxes.size() && cls_shape[0] == boxes.size());
    assert(box_shape[1] == 4 && cls_shape[1] == 2);
    if(points.size() > 0)
    {
        assert(points_shape[0] == boxes.size() && points_shape[1] == 10);
    }

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

void FaceDetector::nms_cpu(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes)
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

void FaceDetector::GetOnnxModelInputInfo(Ort::Session &session_net;, std::vector<const char*> &input_node_names, std::vector<const char*> &output_node_names)
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

        allocator.Free(input_name);
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

        allocator.Free(output_name);
    }

}

int64_t FaceDetector::GetOnnxModelInfo(std::vector<string> model_dir)
{
    if (model_dir.empty())
    {
        printf("model_dir empty, please check it!!!\n");
        return YKX_PARAM_ERROR;
    }

    //GetModel dir
    m_pModel_dir = model_dir[0];
    m_rModel_dir = model_dir[1];
    m_oModel_dir = model_dir[2];

    /* load three networks */
    //p
    session_PNet(env, m_pModel_dir.c_str(), session_option);
    //R
    session_RNet(env, m_rModel_dir.c_str(), session_option);
    //O
    session_ONet(env, m_oModel_dir.c_str(), session_option);

    GetOnnxModelInputInfo(session_PNet, m_PNetInputNodeNames, m_PNetOutputNodeNames);
    GetOnnxModelInputInfo(session_RNet, m_RNetInputNodeNames, m_RNetOutputNodeNames);
    GetOnnxModelInputInfo(session_ONet, m_ONetInputNodeNames, m_ONetOutputNodeNames);

    return YKX_SUCCESS;
}

//#define IMAGE_DEBUG
vector< FaceDetector::BoundingBox > FaceDetector::Detect(const cv::Mat& img, const COLOR_ORDER color_order, const IMAGE_DIRECTION orient, \
                                                            int min_size, float P_thres, float R_thres, float O_thres, bool is_fast_resize,\
                                                             float scale_factor)
{
    /*change image format*/
    cv::Mat sample;
    if( img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        if( color_order == RGBA)
            cv::cvtColor(img, sample, cv::COLOR_RGBA2RGB);
        else
            cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if ( img.channels() == 1 && num_channels_ == 3 )
        cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
    else
        sample = img;
    cv::Mat sample_normalized;
    //convert to float and normalize
    sample.convertTo( sample_normalized, CV_32FC3, img_var, -img_mean * img_var);

    if(orient == IMAGE_DIRECTION::ORIENT_UP)
        sample_normalized = sample_normalized.t();
    else if(orient == IMAGE_DIRECTION::ORIENT_DOWN){
        cv::flip(sample_normalized, sample_normalized, -1);
        sample_normalized = sample_normalized.t();
    }
    else if(orient == IMAGE_DIRECTION::ORIENT_LEFT)
    {
        cv::flip(sample_normalized, sample_normalized, -1);
    }

    vector<float> points;
    const int img_H = sample_normalized.rows;
    const int img_W = sample_normalized.cols;
    int minl  = cv::min(img_H, img_W);
    //split the input image
    vector<cv::Mat> sample_norm_channels;
    cv::split(sample_normalized, sample_norm_channels);
    if(color_order == BGR || color_order == BGRA)
    {
        cv::Mat tmp = sample_norm_channels[0];
        sample_norm_channels[0] = sample_norm_channels[2];
        sample_norm_channels[2] = tmp;
    }

    float m = 12.0 / min_size;
    minl *= m;
    vector<float> all_scales;
    float cur_scale = 1.0;
    while( minl >= 12.0 )
    {
        all_scales.push_back( m * cur_scale);
        cur_scale *= scale_factor;
        minl *= scale_factor;
    }
    /*stage 1: P_Net forward can get rectangle and regression */
    vector<BoundingBox> totalBoxes;
    for(int i = 0; i < all_scales.size(); i ++)
    {
        vector<cv::Mat> pyr_channels;
        cur_scale = all_scales[i];
        int hs = cvCeil(img_H * cur_scale);
        int ws = cvCeil(img_W * cur_scale);
        //对输入的形状进行变化
        size_t input_tensor_size = 1 * hs * ws * 3;
        std::vector<float> input_image_(input_tensor_size);
        std::array<int64_t, 4> input_shape_{ 1, 3, hs, ws };
        Ort::Value input_tensor_pnet = Ort::Value::CreateTensor<float>(memory_info, 
                                                                    input_image_.data(), 
                                                                    input_image_.size(), 
                                                                    input_shape_.data(), 
                                                                    input_shape_.size());
#if 0
        Blob<float>* input_layer = P_Net->input_blobs()[0];
        input_layer->Reshape(1, num_channels_, hs, ws);
        //// forward dimension change to all layers
        P_Net->Reshape();
        //wrap input layers
        wrapInputLayer( P_Net, &pyr_channels);
        //对图像每个通道进行下采样
        pyrDown(sample_norm_channels, cur_scale, &pyr_channels);
        //P Net forward operation
        const vector<Blob<float>*> out = P_Net->Forward();
        /* copy the output layer to a vector*/
        Blob<float>* output_layer0 = out[0];
        vector<int> box_shape = output_layer0->shape();
        int output_size = box_shape[0] * box_shape[1] * box_shape[2] * box_shape[3];
        const float* begin0 = output_layer0->cpu_data();
#endif
        const float* end0 = output_size + begin0;
        vector<float> regs(begin0, end0);
        
        Blob<float>* output_layer1 = out[1];
        vector<int> cls_shape = output_layer1->shape();
        output_size = cls_shape[0] * cls_shape[1] * cls_shape[2] * cls_shape[3];
        const float* begin1 = output_layer1->cpu_data();
    
        const float* end1 = output_size + begin1;
        vector<float> cls(begin1, end1);
        vector<BoundingBox> filterOutBoxes;
        vector<BoundingBox> nmsOutBoxes;
        //vector<BoundingBox> filterOutRegs;
        generateBoundingBox(regs, box_shape, cls, cls_shape, cur_scale, P_thres, filterOutBoxes);
        nms_cpu(filterOutBoxes, 0.5, UNION, nmsOutBoxes);
        if(nmsOutBoxes.size() > 0)
            totalBoxes.insert(totalBoxes.end(), nmsOutBoxes.begin(), nmsOutBoxes.end());
    }

    //do global nms operator
    if (totalBoxes.size() > 0)
    {
        vector<BoundingBox> globalFilterBoxes;
        //cout<<totalBoxes.size()<<endl;
        nms_cpu(totalBoxes, 0.7, UNION, globalFilterBoxes);
        //cout<<globalFilterBoxes.size()<<endl;
        totalBoxes.clear();
        cout<<totalBoxes.size()<<endl;
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

    //the second stage: R-Net
    if(totalBoxes.size() > 0)
    {
        vector<cv::Mat> n_channels;
        Blob<float>* input_layer = R_Net->input_blobs()[0];
        input_layer->Reshape(totalBoxes.size(), num_channels_, 24, 24);
        R_Net->Reshape();
        wrapInputLayer(R_Net, &n_channels);
        //fillout n_channels
        buildInputChannels(sample_norm_channels, totalBoxes, cv::Size(24,24), &n_channels);
        //R_Net forward
        R_Net->Forward();
        /*copy output layer to vector*/
        Blob<float>* output_layer0 = R_Net->output_blobs()[0];
        vector<int> box_shape = output_layer0->shape();
        int output_size = box_shape[0] * box_shape[1];
        const float* begin0 = output_layer0->cpu_data();
        const float* end0 = output_size + begin0;
        vector<float> regs(begin0, end0);
        
        Blob<float>* output_layer1 = R_Net->output_blobs()[1];
        vector<int> cls_shape = output_layer1->shape();
        output_size = cls_shape[0] * cls_shape[1];
        const float* begin1 = output_layer1->cpu_data();
        const float* end1 = output_size + begin1;
        vector<float> cls(begin1, end1);

        vector<BoundingBox> filterOutBoxes;
        filteroutBoundingBox(totalBoxes, regs, box_shape, cls, cls_shape, vector<float>(), vector<int>(), R_thres, filterOutBoxes);
        nms_cpu(filterOutBoxes, 0.7, UNION, totalBoxes);
    }

    // do third stage: O-Net
    if(totalBoxes.size() > 0)
    {
        vector<cv::Mat> n_channels;
        Blob<float>* input_layer = O_Net->input_blobs()[0];
        input_layer->Reshape(totalBoxes.size(), num_channels_, 48, 48);
        O_Net->Reshape();
        wrapInputLayer(O_Net, &n_channels);
        //fillout n_channels
        buildInputChannels(sample_norm_channels, totalBoxes, cv::Size(48,48), &n_channels);
        //O_Net forward
        O_Net->Forward();
        /*copy output layer to vector*/
        Blob<float>* output_layer0 = O_Net->output_blobs()[0];
        vector<int> box_shape = output_layer0->shape();
        int output_size = box_shape[0] * box_shape[1];
        const float* begin0 = output_layer0->cpu_data();
        const float* end0 = output_size + begin0;
        vector<float> regs(begin0, end0);
        
        Blob<float>* output_layer1 = O_Net->output_blobs()[1];
        vector<int> points_shape = output_layer1->shape();
        output_size = points_shape[0] * points_shape[1];
        const float* begin1 = output_layer1->cpu_data();
        const float* end1 = output_size + begin1;
        vector<float> points(begin1, end1);
        
        Blob<float>* output_layer2 = O_Net->output_blobs()[2];
        vector<int> cls_shape = output_layer2->shape();
        output_size = cls_shape[0] * cls_shape[1];
        const float* begin2 = output_layer2->cpu_data();
        const float* end2 = output_size + begin2;
        vector<float> cls(begin2, end2);
        
        vector<BoundingBox> filterOutBoxes;
        filteroutBoundingBox(totalBoxes, regs, box_shape, cls, cls_shape, points, points_shape, O_thres, filterOutBoxes);
        nms_cpu(filterOutBoxes, 0.7, MIN, totalBoxes);
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