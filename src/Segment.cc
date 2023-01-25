/*
 *--------------------------------------------------------------------------------------------------
 * DS-SLAM: A Semantic Visual SLAM towards Dynamic Environments
　*　Author(s):
 * Chao Yu, Zuxin Liu, Xinjun Liu, Fugui Xie, Yi Yang, Qi Wei, Fei Qiao qiaofei@mail.tsinghua.edu.cn
 * Created by Yu Chao@2018.12.03
 * --------------------------------------------------------------------------------------------------
 * DS-SLAM is a optimized SLAM system based on the famous ORB-SLAM2. If you haven't learn ORB_SLAM2 code, 
 * you'd better to be familiar with ORB_SLAM2 project first. Compared to ORB_SLAM2, 
 * we add anther two threads including semantic segmentation thread and densemap creation thread. 
 * You should pay attention to Frame.cc, ORBmatcher.cc, Pointcloudmapping.cc and Segment.cc.
 * 
 *　@article{murORB2,
 *　title={{ORB-SLAM2}: an Open-Source {SLAM} System for Monocular, Stereo and {RGB-D} Cameras},
　*　author={Mur-Artal, Ra\'ul and Tard\'os, Juan D.},
　* journal={IEEE Transactions on Robotics},
　*　volume={33},
　* number={5},
　* pages={1255--1262},
　* doi = {10.1109/TRO.2017.2705103},
　* year={2017}
 *　}
 * --------------------------------------------------------------------------------------------------
 * Copyright (C) 2018, iVip Lab @ EE, THU (https://ivip-tsinghua.github.io/iViP-Homepage/) and 
 * Advanced Mechanism and Roboticized Equipment Lab. All rights reserved.
 *
 * Licensed under the GPLv3 License;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://github.com/ivipsourcecode/DS-SLAM/blob/master/LICENSE
 *--------------------------------------------------------------------------------------------------
 */

#include "Segment.h"
#include "Tracking.h"
#include "Camera.h"
#include <fstream>
#define SKIP_NUMBER 1
using namespace std;

namespace ORB_SLAM2
{
Segment::Segment(const string &torch_model, const string &pascal_png):mbFinishRequested(false),mSkipIndex(SKIP_NUMBER),mSegmentTime(0),imgIndex(0)
{

    model_file = torch_model;//模型文件路径
    
    LUT_file = pascal_png;
    label_colours = cv::imread(LUT_file,1);//读取调色板
    cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);//格式转换
    mImgSegmentLatest=cv::Mat(Camera::height,Camera::width,CV_8UC1);//初始化语义掩膜
    mbNewImgFlag=false;
    img_size=224;//输入模型的图片大小应该为224*224

}

void Segment::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;//Tracking线程指针
}

bool Segment::isNewImgArrived()
{
    unique_lock<mutex> lock(mMutexGetNewImg);//锁线程
    if(mbNewImgFlag)//有新图像，标志重置，返回true
    {
        mbNewImgFlag=false;
        return true;
    }
    else
    return false;
}

torch::Tensor Segment::process(cv::Mat& image,torch::Device device,int img_size){
        
        //首先对输入的图片进行处理
        cv::cvtColor(image, image, CV_BGR2RGB);// bgr -> rgb
        std::cout<<"process start"<<std::endl;
        cv::Mat img_float;
        cv::resize(image, img_float, cv::Size(img_size, img_size));//将输入图片resize为224*224
        std::vector<int64_t> dims = {1, img_size, img_size, 3};//设置输入张量的大小
        torch::Tensor img_var = torch::from_blob(img_float.data, dims, torch::kByte).to(device);//将图像转化成张量
        img_var = img_var.permute({0,3,1,2});//将张量的参数顺序转化为 torch输入的格式 1,3,384,384
        img_var = img_var.toType(torch::kFloat);//
        img_var = img_var.div(255);
 
        return img_var;

    }

void Segment::Run()
{
    std::cout<<"Segment is running"<<endl;
    //设置device类型
    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);
    std::cout<<"cuda support:"<<(torch::cuda::is_available()?"ture":"false")<<std::endl;
    //读取模型
    torch::jit::script::Module module= torch::jit::load(model_file);
    module.to(device);//将模型送入device
    cout << "Load model ..."<<endl;
    while(1)
    {

        usleep(1);
        if(!isNewImgArrived())//检测是否有新的一帧抵达
        continue;

        cout << "Wait for new RGB img time =" << endl;
        if(mSkipIndex==SKIP_NUMBER)//跳过第0帧
        {
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            // Recognise by Semantin segmentation
            torch::Tensor img_var=process(mImg,device,img_size);//将输入图像转换为Tensor
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            torch::Tensor result = module.forward({img_var}).toTensor();  //前向传播获取结果，还是tensor类型
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0 << " sec" <<std::endl;
            result=result.argmax(1);//找出每个点概率最大的一个
            result = result.squeeze();//删除一个维度   
            result = result.to(torch::kU8);//这里是为了让分割的区域更明显,但是不需要加，因为后面使用了LUT的方法可以使不同的mask显示不同颜色
            result=result.to(torch::kCPU);//将数据送入CPU
            cv::Mat pts_mat(cv::Size(224,224), CV_8U, result.data_ptr());//新建一个矩阵，用于保存数据，将tensor的数据转移到这里面
            
            mImgSegment=pts_mat.clone();//为mImgSegment赋值
            mImgSegment_color = mImgSegment.clone();//为mImgSegment_color赋值
            cv::cvtColor(mImgSegment,mImgSegment_color, CV_GRAY2BGR);//将灰度图转换为三通道的RGB图

            LUT(mImgSegment_color, label_colours, mImgSegment_color_final);//上色
            cv::resize(mImgSegment, mImgSegment, cv::Size(Camera::width,Camera::height) );
            cv::resize(mImgSegment_color_final, mImgSegment_color_final, cv::Size(Camera::width,Camera::height) );
            cv::imshow( "Display window", mImgSegment_color_final);

            std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            mSegmentTime+=std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
            mSkipIndex=0;
            imgIndex++;
        }
        mSkipIndex++;
        // 保存了上一帧 和本帧的 分割图片mImgSegmentLatest  ，并且告诉Track 新的分割图片已经就绪
        ProduceImgSegment();
        if(CheckFinish())//检测是否结束分割线程
        {
            break;
        }

    }

}

bool Segment::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);//锁线程
    return mbFinishRequested;
}
  
void Segment::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);//锁线程
    mbFinishRequested=true;//线程结束标识符置true
}

void Segment::ProduceImgSegment()
{
    std::unique_lock <std::mutex> lock(mMutexNewImgSegment);//锁线程
    mImgTemp=mImgSegmentLatest;
    mImgSegmentLatest=mImgSegment;
    mImgSegment=mImgTemp;
    mpTracker->mbNewSegImgFlag=true;
}

}


