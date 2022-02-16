#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h> 
#include <Eigen/Core>
#include <matplot/matplot.h>
#include <Eigen/LU>


using namespace cv;
using namespace std;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix;

#ifdef CUDA_NOT_FOUND
    #include "helper.cu"
#else
    // #include "helper.cu"
#endif // #ifdef CUDA_NOT_FOUND

Vector3d solve(MatrixXd A, VectorXd b){
    MatrixXd At = A.transpose();
    MatrixXd AAt = At*A; 
    MatrixXd AAtI = AAt.inverse();
    Vector3d coeff = AAtI*(At*b);
    return coeff;
}

void graph(Vector3d x, vector<KeyPoint>& trajectory, vector<KeyPoint>& keypts){    
    float fpt = keypts.front().pt.x;
    float bpt = keypts.back().pt.x;
    for(int i=fpt; i<bpt; i++){

        float val = x(0)*i*i + x(1)*i+ x(2);

        KeyPoint* kpt=new KeyPoint();
        // cout<<val<<endl;
        kpt->pt.x=i;
        kpt->pt.y = val;
        keypts.push_back(*kpt);
    }
}

int main(int argc, char** argv){
    VideoCapture capture;
    Mat frame;
    vector<double> A_vec;
    vector<double> b_vec;

    capture.open("/home/aswath/umd/673/ENPM-673/Homework1/aswath_hw1/cpp/ball_video2.mp4", cv::CAP_ANY);
    if(capture.isOpened()){
        cout<<"Error opening the file"<<endl;
    }

    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 300;

    // Filter by Circularity
    params.filterByCircularity = false;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = false;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params); 
        Mat im_with_keypoints;

    vector<KeyPoint> keypts;
    while(true){
        capture.read(frame);
        if(frame.empty()){
            break;
        }
        std::vector<KeyPoint> keypointsPframe;
        cv::resize(frame, frame, cv::Size(), 0.25, 0.25);
        detector->detect(frame, keypointsPframe);
        keypts.push_back(keypointsPframe[0]);
        float pt_x = keypointsPframe[0].pt.x;
        float pt_y = keypointsPframe[0].pt.y;
        A_vec.push_back(pt_x);
        b_vec.push_back(pt_y);
        drawKeypoints(frame, keypts, im_with_keypoints, Scalar(255,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        cv::imshow("Trajectory", im_with_keypoints);// Show blobs
        waitKey(1);
    }

    int datapoints = A_vec.size();
    cout<<"No. of datapoints: "<<datapoints;
    MatrixXd A(datapoints,3);
    VectorXd b(datapoints);

    for(int i=0;i<datapoints;i++){
        A.row(i)<<A_vec[i]*A_vec[i], A_vec[i], 1;
    }
    for(int i=0;i<datapoints;i++){
        b(i)=b_vec[i];
    }

    Vector3d x;
    x=solve(A,b);
    cout<<"\nX: "<<x<<"Size: "<<x.size();
    std::vector<KeyPoint> trajectory;
    graph(x,trajectory,keypts);
    int myradius=5;
    for (int i=0;i<keypts.size();i++){
        try{
            circle(im_with_keypoints,cv::Point(keypts[i].pt.x, keypts[i].pt.y),myradius,CV_RGB(100,0,0),-1,8,0);
        }catch(cv::Exception e){}
    }

    cv::imshow("Trajectory", im_with_keypoints);
    waitKey(10000);

    return 0;
}