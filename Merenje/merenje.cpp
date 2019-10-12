#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include "../Core/core.cpp"

using namespace std;
using namespace cv;

#define PI 3.14159265

const float calibrationSquareDimension = 1.0;
const Size chessBoardDimension = Size(9, 6);
// Neophodne velicine
Mat cameraMatrix(3, 3, CV_32F);
Mat distortionCoeff(5, 1, CV_32F);

Mat rodrigez(Mat rvec)
{

    float theta = sqrt(rvec.at<float>(0, 0) * rvec.at<float>(0, 0) + rvec.at<float>(1, 0) * rvec.at<float>(1, 0) + rvec.at<float>(2, 0) * rvec.at<float>(2, 0));
    Mat vec(3, 1, CV_32F);
    vec.at<float>(0, 0) = rvec.at<float>(0, 0) / theta;
    vec.at<float>(1, 0) = rvec.at<float>(1, 0) / theta;
    vec.at<float>(2, 0) = rvec.at<float>(2, 0) / theta;
    float ux = vec.at<float>(0, 0);
    float uy = vec.at<float>(1, 0);
    float uz = vec.at<float>(2, 0);
    Mat rmat(3, 3, CV_32F);
    rmat.at<float>(0, 0) = cos(theta) + ux * ux * (1 - cos(theta));
    rmat.at<float>(0, 1) = ux * uy * (1 - cos(theta)) - uz * sin(theta);
    rmat.at<float>(0, 2) = ux * uz * (1 - cos(theta)) + uy * sin(theta);
    rmat.at<float>(1, 0) = uy * ux * (1 - cos(theta)) + uz * sin(theta);
    rmat.at<float>(1, 1) = cos(theta) + uy * uy * (1 - cos(theta));
    rmat.at<float>(1, 2) = uy * uz * (1 - cos(theta)) - ux * sin(theta);
    rmat.at<float>(2, 0) = uz * ux * (1 - cos(theta)) - uy * sin(theta);
    rmat.at<float>(2, 1) = uz * uy * (1 - cos(theta)) + ux * sin(theta);
    rmat.at<float>(2, 2) = cos(theta) + uz * uz * (1 - cos(theta));
    return rmat;
}

bool loadCameraCalibration()
{
    FileStorage fs("file.yml", FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoeff"] >> distortionCoeff;
    cameraMatrix.convertTo(cameraMatrix, CV_32F);
    distortionCoeff.convertTo(distortionCoeff, CV_32F);
    fs.release();
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f> &corners)
{
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            corners.push_back(Point3f((float)(j * squareEdgeLength), (float)(i * squareEdgeLength), 0.0));
        }
    }
}

void getChessBoardCorners(Mat image, vector<Point2f> &allFoundCorners, bool showResult = false)
{
    bool found = findChessboardCorners(image, chessBoardDimension, allFoundCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
    Mat tmp = image;
    if (!found)
    {
        printf("Chess board missing!\n");
    }
    drawChessboardCorners(image, chessBoardDimension, allFoundCorners, found);
    imshow("ChessBoard", image);
    waitKey(0);
}


void getRelativeCameraLocation(Mat image, Mat &tvec, Mat &rmat)
{

    vector<Point2f> checkerboardImageSpacePoints;
    getChessBoardCorners(image, checkerboardImageSpacePoints, false);
    vector<Point2f> checkerboardImageSpacePointsinNewCS;

    vector<Point3f> worldSpaceCornerPoints;
    createKnownBoardPosition(chessBoardDimension, calibrationSquareDimension, worldSpaceCornerPoints);

    vector<int> inliers;
    Mat rvec;
    solvePnPRansac(worldSpaceCornerPoints, checkerboardImageSpacePoints, cameraMatrix, distortionCoeff, rvec, tvec, false, 100, 1, worldSpaceCornerPoints.size() / 2, inliers, CV_ITERATIVE);
    //solvePnP(worldSpaceCornerPoints, checkerboardImageSpacePoints, cameraMatrix, distortionCoeff, rvec, tvec, true, CV_ITERATIVE);

    tvec.convertTo(tvec, CV_32F);
    rvec.convertTo(rvec, CV_32F);
    rmat = rodrigez(rvec);
    //Rodrigues(rvec, rmat);
    FileStorage fs("Vektori.yml", FileStorage::WRITE);
    fs << "RMAT" << rmat;
    fs << "RVEC" << rvec;
    fs << "TVEC" << tvec;
    fs.release();
}

int main(int argc, char **argv)
{
    Mat frame;
    Mat frameCorners;

    loadCameraCalibration();

    /*VideoCapture vid(1);

    if(!vid.isOpened())
    {
        return 0;
    }

    int framesPerSecond = 30;
    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

    Mat img;

    int imgs = 1;
    int count = 0;

    Mat loadimg1, loadimg2, loadimg3;
    while(true)
    {
         if(!vid.read(frame))
         {
             break;
         }
         vector<Vec2f> foundPoints;
         bool found = findChessboardCorners(frame, chessBoardDimension, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
         frame.copyTo(frameCorners);
         drawChessboardCorners(frameCorners, chessBoardDimension, foundPoints, found);

         if(found)
          {
             imshow("Webcam", frameCorners);
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
                imwrite("Images//img" + to_string(imgs) + "-" + to_string(count) + ".jpg",img);
                count++;
                if(count == 1)
                {
                    getCameraPosition(img, tvec, rmat);

                }
                if(count == 3)
                {
                    Mat loadimg1 = imread("Images//img" + to_string(imgs) + "-2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
                    Mat loadimg2 = imread("Images//img" + to_string(imgs) + "-4.jpg", CV_LOAD_IMAGE_GRAYSCALE);
                    findPlane(loadimg1, loadimg2);
                    count = 0;
                    imgs++;
                }
                break;
             case 27:
                //exit
                cout << "Finish \n";
                return 0;
                break;
              case 'q':
                registerImages();
                printf("Registred \n");
                break;
         }
    }*/
    //namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
    Mat loadimg1 = imread("Images//img1-0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat loadimg2 = imread("Images//img1-0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat loadimg3 = imread("Images//img1-2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cout << "Resolution :\t" << loadimg1.size() << "\n";
    /*Mat filtered1, filtered2, filtered3;
    bilateralFilter(loadimg1, filtered1, 5, 40, 40);
    bilateralFilter(loadimg2, filtered2, 5, 40, 40);
    bilateralFilter(loadimg3, filtered3, 5, 40, 40);
    imshow("NonFiltered", loadimg1);
    imshow("Filtered", filtered1);
    waitKey(0);*/
    Mat img1, img2, img3;
    undistort(loadimg1, img1, cameraMatrix, distortionCoeff);
    undistort(loadimg2, img2, cameraMatrix, distortionCoeff);
    undistort(loadimg3, img3, cameraMatrix, distortionCoeff);
    //imshow("Webcam", img3);
    //waitKey(0);

    vector<Point2f> correctMatchingPoints1;
    vector<Point2f> correctMatchingPoints2;
    Mat_<float> fundamentalMatrix;
    float error = findFeaturePoints(img2, img3, correctMatchingPoints1, correctMatchingPoints2, fundamentalMatrix);
    if (correctMatchingPoints1.size() == 0 || correctMatchingPoints1.size() == 0)
    {
        cout << "\nCan't find feature points\n";
        return 0;
    }
    cout << "FundamentalMat error \t" << error << "\n";

    vector<Point3f> points3dfirst;
    vector<Point3f> points3dsecond;
    vector<float> plane = getPlane(cameraMatrix, distortionCoeff, fundamentalMatrix, correctMatchingPoints1, correctMatchingPoints2, points3dfirst, points3dsecond);
    if (plane.size() == 0)
    {
        cout << "\nCan't fit plane\n";
        return 0;
    }
    showFeaturePoints(img2, img3, correctMatchingPoints1, correctMatchingPoints2, fundamentalMatrix);

    /*float check = checkResults(img2, cameraMatrix, points3dfirst, correctMatchingPoints1);
    cout << "\nCheck " << check << "\n";*/

    cout << "\nRegistration ... \n";
    Mat tvec, rmat;
    getRelativeCameraLocation(img1, tvec, rmat);
    vector<float> chessboardmodel{0, 0, 1, 0};
    //vector<float> chessboard = transformPlane(rmat, tvec, chessboardmodel, true);
    vector<Point3f> chessboardPointsmodel;
    createKnownBoardPosition(chessBoardDimension, calibrationSquareDimension, chessboardPointsmodel);
    vector<Point3f> chessboardPoints = transformPoints(rmat, tvec, chessboardPointsmodel, true);
    int k = 100;
    vector<float> chessboard = ransac(chessboardPoints, k, 1, 3 * chessboardPoints.size() / 4);

    float angle = getAngle(chessboard, plane);
    cout << "\nAngle between planes = \t" << angle << " degrees \n";
    cout << "Finished \n";

    FileStorage fs("Results.yml", FileStorage::WRITE);
    fs << "Plane" << plane;
    fs << "Chessboard" << chessboard;
    fs << "PlanePoints" << points3dfirst;
    fs << "ChessboardPoints" << chessboardPoints;
    fs.release();

    return 0;
}
