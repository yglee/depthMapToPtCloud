#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

void pixelToVector(const cv::Point2i &pixel, const cv::Point2i &centerOfProj, 
		   const cv::Point2f &focalLen, cv::Mat &ray) {
   
    ray.at<float>(0,0) = (pixel.x - centerOfProj.x) / focalLen.x;
    ray.at<float>(1,0) = (pixel.y - centerOfProj.y) / focalLen.y;
    ray.at<float>(2,0) = 1;
    //we need a unit ray
    ray = ray * (1.0 /norm(ray));  
}

int main( int argc, char** argv )
{
    Mat image;
    image = imread( argv[1], CV_LOAD_IMAGE_ANYDEPTH);
 
    if( argc != 2 || !image.data ) {	
	printf( "No image data \n" );
	return -1;
    }
    cout<<"rows: "<<image.rows<<" cols:"<<image.cols<<endl;

    //algorithm: pixel to unit vector, multiply unit vector by depth, add the center of proj.
    unsigned short depth;
    Mat ray(3,1,CV_32F); //ray from center of projection to pixel in 2D image coordinate.
    Point2i pixelCoord; //point in 2D image coordinate
    Point2i centerOfProj = Point2i(0,0); //center of projection
    Point2f focalLen = Point2f(743.2, 742.1); //focal length	    
    Point3f dataPt; //data point in point cloud

    //pointCloud buffer (vector of point3fs)
    std::vector<Point3f> ptCloud;
    
    //write out the points to file
    ofstream ptCloudFile;
    ptCloudFile.open("ptcloud-output.txt");
 
    for (size_t y=0; y < image.rows; y++) {
	for (size_t x=0; x< image.cols; x++) {
	    depth = image.at<ushort>(y,x); //depth is a 16 bit value stored in each pixel
	    pixelCoord = Point2i(x,y); 
	    pixelToVector(pixelCoord,centerOfProj, focalLen, ray);
	    Mat temp = ray*depth;
	    dataPt = Point3f(temp.at<float>(0,0)+centerOfProj.x, 
			     temp.at<float>(1,0)+centerOfProj.y, 
			     temp.at<float>(2,0));  
	    //cout<<dataPt.x<<" "<<dataPt.y<<" "<<dataPt.z<<endl; 
	    ptCloud.push_back(dataPt); 
	    ptCloudFile<<dataPt.x<<" "<<dataPt.y<<" "<<dataPt.z<<endl; 
	}
    }
    ptCloudFile.close();
    return 0;
}
