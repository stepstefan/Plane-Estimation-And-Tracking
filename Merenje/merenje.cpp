#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include "../Final/final.cpp"

using namespace std;
using namespace cv;


const float calibrationSquareDimension = 1;
const Size chessBoardDimension = Size(9, 6);
Mat cameraMatrix;
Mat distortionCoeff;

bool loadCameraCalibration(Mat& cameraMatrix, Mat& distortionCoeff)
{
    FileStorage fs("file.yml", FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoeff"] >> distortionCoeff;
    fs.release();
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3d>& corners)
{
    for(int i = 0; i<boardSize.height; i++)
    {
        for(int j = 0 ; j<boardSize.width; j++)
        {
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
        }
    }
}

void getChessBoardCorners(Mat image, vector<Point2d>& allFoundCorners, bool showResult = false)
{
        bool found = findChessboardCorners(image, Size(9,6), allFoundCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

        if(!found)
        {
            printf("Chess board missing!\n");
        }

}

void getRelativeCameraLocation(Mat image, Mat& rvec, Mat& tvec)
{

    vector<Point2d>  checkerboardImageSpacePoints;
    getChessBoardCorners(image, checkerboardImageSpacePoints, false);

    vector<Point3d> worldSpaceCornerPoints;


    createKnownBoardPosition(chessBoardDimension, calibrationSquareDimension, worldSpaceCornerPoints);

    //worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints);

    solvePnP(worldSpaceCornerPoints, checkerboardImageSpacePoints, cameraMatrix, distortionCoeff, rvec, tvec);
}

int main(int argc, char** argv)
{
    Mat frame;
    Mat drawToFrame;

    loadCameraCalibration(cameraMatrix, distortionCoeff);

    VideoCapture vid(1);

    if(!vid.isOpened())
    {
         return 0;
    }

    int framesPerSecond = 30;


    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

    Mat img;

    int imgs = 1;
    int pair = 1;


    Mat rvec;
    Mat tvec;

    while(true)
    {
         if(!vid.read(frame))
         {
             break;
         }
         vector<Vec2f> foundPoints;
         bool found = false;

         //found = findChessboardCorners(frame, chessBoardDimension, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
         //frame.copyTo(drawToFrame);
         //drawChessboardCorners(drawToFrame, chessBoardDimension, foundPoints, found);
         if(found)
         {
             imshow("Webcam", drawToFrame);
         }
         else
         {
             imshow("Webcam", frame);
         }

         char character = waitKey(100 / framesPerSecond);
         switch(character)
         {
             case ' ':
                img = frame;
                imwrite("Images//img" + to_string(imgs) + "-" + to_string(pair) + ".jpg",img);
                if(pair == 1)
                {
                    getRelativeCameraLocation(img,rvec,tvec);

                    FileStorage fs("Vektori//vektori" + to_string(imgs) + ".yml", FileStorage::WRITE);
                    fs << "rvec" << rvec;
                    fs << "tvec" << tvec;
                    fs.release();
                }
                pair++;
                if(pair == 4)
                {
                    Mat loadimg1 = imread("Images//img" + to_string(imgs) + "-2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
                    Mat loadimg2 = imread("Images//img" + to_string(imgs) + "-3.jpg", CV_LOAD_IMAGE_GRAYSCALE);

                    vector<Point2f> correctMatchingPoints1, correctMatchingPoints2;
            		vector<KeyPoint> correctKeyPoints1, correctKeyPoints2;
            		vector<DMatch> correctMatches;
            		vector<Point3f> points3d;
                    vector<float> plane = getPlane(loadimg1, loadimg2, cameraMatrix, distortionCoeff,correctMatchingPoints1,correctMatchingPoints2,correctKeyPoints1,correctKeyPoints2,correctMatches,points3d);

                    Mat rmat;
            		Rodrigues(rvec, rmat);
                    FileStorage store("Results//result"+ to_string(imgs) +".txt", FileStorage::WRITE);
                    store << "Rotation before Transpose" << rmat;
                    store << "Translation" << tvec;
                    vector<float> newPlane = transformPlane(rmat,tvec,plane);

                    store << "Plane" << plane;
                    store << "NewPlane" << newPlane;
                    store.release();

                    pair = 1;
                    imgs++;
                }
                break;
             case 27:
                //exit
                return 0;
                break;
         }
    }

    return 0;
}
