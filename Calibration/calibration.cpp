#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;

const float calibrationSquareDimension = 1;
const Size chessBoardDimension = Size(9, 6);

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
    for(int i = 0; i<boardSize.height; i++)
    {
        for(int j = 0 ; j<boardSize.width; j++)
        {
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
        }
    }
}

void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f> >& allFoundCorners, bool showResult = false)
{
    for(vector<Mat>::iterator i = images.begin(); i != images.end(); i++)
    {
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(*i, Size(9,6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

        if(found)
        {
            allFoundCorners.push_back(pointBuf);
        }

        if(showResult)
        {
            drawChessboardCorners(*i, Size(9,6), pointBuf, found);
            imshow("", *i);
            waitKey(0);
        }
    }
}
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distortionCoeff, double& calibError)
{
    vector<vector<Point2f> > checkerboardImageSpacePoints;
    getChessBoardCorners(calibrationImages, checkerboardImageSpacePoints, false);

    vector<vector<Point3f> > worldSpaceCornerPoints(1);


    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);

    worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors;
    distortionCoeff = Mat::zeros(8,1, CV_64F);


    calibError = calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distortionCoeff, rVectors, tVectors);
    printf("%f", calibError);
}

bool saveCameraCalibration(Mat cameraMatrix, Mat distortionCoeff, double calibError)
{
    FileStorage fs("file.yml", FileStorage::WRITE);
    fs << "cameraMatrix" << cameraMatrix;
    fs << "distortionCoeff" << distortionCoeff;
    fs << "calibError" << calibError;
    fs.release();


    //_______________________________________________________________________________________________________________________________________________
    ofstream outStream("Rezultati.txt");
    if(outStream)
    {
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;

        for(int r = 0; r<rows; r++)
        {
            for(int c = 0; c < columns; c++)
            {
                double value = cameraMatrix.at<double>(r,c);
                outStream << value << endl;
            }
        }

        rows = distortionCoeff.rows;
        columns = distortionCoeff.cols;

        for(int r = 0; r<rows; r++)
        {
            for(int c = 0; c < columns; c++)
            {
                double value = distortionCoeff.at<double>(r,c);
                outStream << value << endl;
            }
        }
        outStream.close();
        return true;
    }
    return false;
}


int main(int argc, char** argv)
{
    Mat frame;
    Mat drawToFrame;

    Mat cameraMatrix = Mat::eye(3,3, CV_64F);

    Mat distortionCoeff;
    double calibError;
    vector<Mat> savedImages;
    vector<vector<Point2f> > markerCorners, rejectedCandidates;

    VideoCapture vid(1);

    /*Mat temp = imread("/images/left01.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left02.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left03.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left04.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left05.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left06.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left07.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left08.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left09.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left10.jpg", 0);
    savedImages.push_back(temp);
    temp = imread("/images/left11.jpg", 0);
    savedImages.push_back(temp);

    cameraCalibration(savedImages, chessBoardDimension, calibrationSquareDimension, cameraMatrix, distortionCoeff);
    saveCameraCalibration(cameraMatrix, distortionCoeff);*/

   if(!vid.isOpened())
    {
        return 0;
    }

    int framesPerSecond = 20;


    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

    while(true)
    {
        if(!vid.read(frame))
        {
            break;
        }

        vector<Vec2f> foundPoints;
        bool found = false;

        found = findChessboardCorners(frame, chessBoardDimension, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessBoardDimension, foundPoints, found);
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
                //saving
                if(found)
                {
                    Mat temp;
                    frame.copyTo(temp);
                    savedImages.push_back(temp);
                }

                break;
            case 'f':
                //calibration
                if(savedImages.size()>1)
                {
                    cameraCalibration(savedImages, chessBoardDimension, calibrationSquareDimension, cameraMatrix, distortionCoeff, calibError);
                    saveCameraCalibration(cameraMatrix, distortionCoeff, calibError);
                    printf("File created");
                }
                break;
            case 27:
                //exit
                return 0;
                break;
        }


    }
}
