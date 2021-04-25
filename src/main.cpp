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
#include "facedetector.h"

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>


int main(int argc, char **argv)
{
    vector<string> model_path{"/mnt/share/code/onnxruntime_cpp/bin/optimaizer_pnet.onnx", 
                            "/mnt/share/code/onnxruntime_cpp/bin/optimaizer_rnet.onnx",
                            "/mnt/share/code/onnxruntime_cpp/bin/optimaizer_onet.onnx"};
     FaceDetector fd(model_path);
    
    vector<string> imgList;
    readFileList("/mnt/share/code/onnxruntime_cpp/bin/face/", imgList);
    for(int i = 0; i < imgList.size(); i ++)
    {
        cv::Mat testImg = cv::imread(imgList[i]);
        
        vector<FaceDetector::BoundingBox> res = fd.Detect(testImg, FaceDetector::BGR, FaceDetector::ORIENT_UP ,20, 0.6, 0.7, 0.7);
        cout<< "detected face NUM : " << res.size() << endl;
        for(int k = 0; k < res.size(); k++)
        {
            cv::rectangle(testImg, cv::Point(res[k].x1, res[k].y1), cv::Point(res[k].x2, res[k].y2), cv::Scalar(0, 255, 255), 2);
            for(int i = 0; i < 5; i ++)
                cv::circle(testImg, cv::Point(res[k].points_x[i], res[k].points_y[i]), 2, cv::Scalar(0, 255, 255), 2);
        }
        string picName = "test" + to_string(i) + ".jpg";
        cv::imwrite(picName, testImg);
    }
    return 0;
}