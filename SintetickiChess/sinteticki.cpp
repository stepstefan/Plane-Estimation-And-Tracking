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

bool loadCameraCalibration()
{
    FileStorage fs("file.yml", FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoeff"] >> distortionCoeff;
    cameraMatrix.convertTo(cameraMatrix, CV_32F);
    distortionCoeff.convertTo(distortionCoeff, CV_32F);
    fs.release();
}

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

vector<Point3f> generateChessboard()
{
    vector<Point3f> chessboardPoints;
    for (size_t i = 0; i < chessBoardDimension.width; i++)
    {
        for (size_t j = 0; j < chessBoardDimension.height; j++)
        {
            Point3f tmp;
            tmp.x = 1000 + 100*i;
            tmp.y = 500 + 100*j;
            tmp.z = 10;
            chessboardPoints.push_back(tmp);
        }
    }
    return chessboardPoints;
}

vector<Point2f> sintetickiChessboard(Mat cameraMatrix, Mat_<float> wp) // t - translacija kamere
{
	Mat_<float> tmp;
	Mat_<float> res;
    vector<Point2f> points;
	Mat_<float> wpn = wp.clone();
	for (size_t i = 0; i < wp.rows; i++)
	{
		tmp = wp.row(i).t();
		res = cameraMatrix * tmp;
		points.push_back(Point2f(res.at<float>(0, 0) / res.at<float>(2, 0), res.at<float>(1, 0) / res.at<float>(2, 0)));
	}
    return points;
}

int main(int argc, char **argv)
{
    //loadCameraCalibration();
    cameraMatrix = (Mat_<float>(3, 3) << 4, 0, 2,
								0, 4, 2,
								0, 0, 1);
    vector<Point3f> chessboardPoints = generateChessboard();
    vector<Point2f> imageChess;// = sintetickiChessboard(cameraMatrix, Mat(chessboardPoints));
    Vec3f r(0, 0, 0); //Rodrigues(R ,rvec);
	Vec3f t(0, 0, 0);
    projectPoints(chessboardPoints, r, t, cameraMatrix, Mat(), imageChess);
    //cout << Mat(imageChess) << "\n";
    vector<Point3f> worldSpaceCornerPoints;
    createKnownBoardPosition(chessBoardDimension, calibrationSquareDimension, worldSpaceCornerPoints);
    //cout << Mat(worldSpaceCornerPoints).depth();
    //cout << Mat(worldSpaceCornerPoints) << "\n";
    vector<u_char> inliers;
    Mat rvec, rmat, tvec;
    solvePnP(worldSpaceCornerPoints, imageChess, cameraMatrix, Mat(), rvec, tvec, false, CV_ITERATIVE);
    cout << rvec << "\n";
    cout << tvec << "\n";
    solvePnP(worldSpaceCornerPoints, imageChess, cameraMatrix, Mat(), rvec, tvec, true, CV_ITERATIVE);
    //cout << rvec << "\n";
    //cout << tvec << "\n";
    solvePnP(worldSpaceCornerPoints, imageChess, cameraMatrix, Mat(), rvec, tvec, true, CV_ITERATIVE);
    //cout << rvec << "\n";
    //cout << tvec << "\n";
    solvePnP(worldSpaceCornerPoints, imageChess, cameraMatrix, Mat(), rvec, tvec, true, CV_ITERATIVE);
    cout << rvec << "\n";
    cout << tvec << "\n";

    //solvePnPRansac(worldSpaceCornerPoints, imageChess, cameraMatrix, Mat(), rvec, tvec, true, 100, 1, worldSpaceCornerPoints.size() / 2, inliers, CV_ITERATIVE);
    cout << rvec << "\n";
    cout << tvec << "\n";

    tvec.convertTo(tvec, CV_32F);
    rvec.convertTo(rvec, CV_32F);
    //rmat = rodrigez(rvec);
    Rodrigues(rvec, rmat);
    /*cout << rmat << "\n";
    cout << tvec << "\n";*/
    vector<Point3f> newchessPoints = transformPoints(rmat, tvec, worldSpaceCornerPoints, true);
    int k = 100;
    vector<float> chessplane{0,0,1,0};
    vector<float> chessboardmodel{0, 0, 1, 0};
    //vector<float> newplane = transformPlane(rmat, tvec, chessboardmodel, false);
    vector<float> newplane = ransac(newchessPoints, k, 0.005, 3*newchessPoints.size()/4);
    /*float tmp = chessplane[3]/newplane[3];
    newplane[0] = newplane[0] * tmp;
    newplane[1] = newplane[1] * tmp;
    newplane[2] = newplane[2] * tmp;
    newplane[3] = chessplane[3];*/
    float error  = getAngle(chessplane, newplane);
    cout << Mat(chessplane) << "\n";
    cout << Mat(newplane) << "\n";
    cout << error << "\n";
    /*for(size_t i = 0; i < newchessPoints.size(); i++)
	{	
		float scale = 10 / newchessPoints[i].z;
        newchessPoints[i].x = scale * newchessPoints[i].x; 
        newchessPoints[i].y = scale * newchessPoints[i].y;
        newchessPoints[i].z = 10;
	}*/
    cout << Mat(newchessPoints) << "\n";
    cout << Mat(chessboardPoints) << "\n";


    return 0;
}
