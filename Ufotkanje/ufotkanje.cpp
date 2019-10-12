#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;

bool loadCameraCalibration(Mat &cameraMatrix, Mat &distortionCoeff)
{
    FileStorage fs("file.yml", FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoeff"] >> distortionCoeff;
    fs.release();
}

int main(int argc, char **argv)
{
    Mat frame;
    Mat cameraMatrix;
    Mat distortionCoeff;

    loadCameraCalibration(cameraMatrix, distortionCoeff);

    VideoCapture vid(0);

    if (!vid.isOpened())
    {
        return 0;
    }

    int framesPerSecond = 20;

    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

    Mat img1;
    Mat img2;
    bool ok = false;
    while (true)
    {
        if (!vid.read(frame))
        {
            break;
        }
        imshow("Webcam", frame);

        char character = waitKey(100 / framesPerSecond);
        switch (character)
        {
        case ' ':
            if (!ok)
            {
                img1 = frame;
                imwrite("img1.jpg", img1);
                ok = true;
            }
            else
            {
                img2 = frame;
                imwrite("img2.jpg", img2);
                ok = false;
            }
            break;
        case 27:
            //exit
            return 0;
            break;
        }
    }
    imshow("Webcam", img1);

    waitKey();

    return 0;
}
