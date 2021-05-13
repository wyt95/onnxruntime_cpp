#include "facedetector.h"

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
            if((strstr(ptr->d_name, "jpg") != NULL) || (strstr(ptr->d_name, "jpeg") != NULL))
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
                cout << "base:" << base << endl;
            }
            cout << "ptr->d_name:" << ptr->d_name << endl;
            cout << "imgFiles:" << imgFiles.size() << endl;
        }
    }
    closedir(dir);
}

int main(int argc, char **argv)
{
    double timeStart = (double)getTickCount();
    vector<string> model_path{"../model/optimaizer_pnet.onnx", 
                            "../model/optimaizer_rnet.onnx",
                            "../model/optimaizer_onet.onnx"};
    cout << "model_path :" << model_path.size() << endl;
    FaceDetector *fd = new FaceDetector(model_path);

    cout << "FaceDetector Init()....." << endl;
    fd->Init();

    double nTime_1 = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "running time init ：" << nTime_1 << "sec\n" << endl;

    vector<string> imgList;
    cout << "start readFileList....." << endl;
    readFileList("../face", imgList);
    cout << "start anylyis....." << endl;
    cout << "imgList size : " << imgList.size() << endl;
    for(int i = 0; i < imgList.size(); i ++)
    {
        cv::Mat testImg = cv::imread(imgList[i]);
        
        vector<FaceDetector::BoundingBox> res = fd->Detect(testImg, FaceDetector::BGR, FaceDetector::ORIENT_UP ,20, 0.6, 0.7, 0.7);
        cout<< "detected face NUM : " << res.size() << endl;
        for(int k = 0; k < res.size(); k++)
        {
            cv::rectangle(testImg, cv::Point(res[k].x1, res[k].y1), cv::Point(res[k].x2, res[k].y2), cv::Scalar(0, 255, 255), 2);
            for(int i = 0; i < 5; i ++)
                cv::circle(testImg, cv::Point(res[k].points_x[i], res[k].points_y[i]), 2, cv::Scalar(0, 255, 255), 2);
        }
        string picName = "../out/test" + to_string(i) + ".jpg";
        cv::imwrite(picName, testImg);
    }

    delete fd;
    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "running time ：" << nTime << "sec\n" << endl;
    cout << "end anylyis....." << endl;
    return 0;
}