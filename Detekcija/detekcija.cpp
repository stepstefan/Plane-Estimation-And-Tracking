#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <random>
#include <time.h>

using namespace std;
using namespace cv;



//Drawing of epipolar line_____________________________________________________________________________________________________________________________________________________________________________________

//* \brief Compute and draw the epipolar lines in two images
 //*      associated to each other by a fundamental matrix
 //*
 //* \param F         Fundamental matrix
 //* \param img1      First image
 //* \param img2      Second image
 //* \param points1   Set of points in the first image
 //* \param points2   Set of points in the second image matching to the first set
 //* \param inlierDistance      Points with a high distance to the epipolar lines are
 //*                not displayed. If it is negative, all points are displayed


//Draw Epipolar Lines_______________________________________________________________________________________________________________________________________________________________________________
float distancePointLine(const Point2f point, const Vec<float,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return fabsf(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

void drawEpipolarLines(cv::Mat& F,
                cv::Mat& img1, cv::Mat& img2,
                std::vector<cv::Point2f> points1,
                std::vector<cv::Point2f> points2,
                float inlierDistance = -1)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

  // * Allow color drawing

  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec<float,3> > epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);

  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());

  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }

    // * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa

    cv::Scalar color(rng(256),rng(256),rng(256));

    Mat tmp2 = outImg(rect2); //non-connst lvalue refernce to type cv::Mat cannot bind to a temporary of type cv::Mat
    Mat tmp1 = outImg(rect1);

    cv::line(tmp2,
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(tmp1, points1[i], 3, color, -1, CV_AA);

    cv::line(tmp1,
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(tmp2, points2[i], 3, color, -1, CV_AA);
  }
  cv::imshow("Epl", outImg);
  cv::waitKey(1);
}

//RANSAC___________________________________________________________________________________________________________________________________________________________________________________________________
float distancePointPlane(const Point3f& point, const vector<float>& plane)
{
      return fabsf(plane[0]*point.x + plane[1]*point.y + plane[2]*point.z + plane[3])
      / sqrt(plane[0]*plane[0] + plane[1]*plane[1] + plane[2] * plane[2]) ;
}

vector<float> getPlaneParams(vector<Point3f>& samplePoints)
{
    vector<float> plane;
    float a = ((samplePoints[0].y - samplePoints[1].y)*(samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].y - samplePoints[2].y)*(samplePoints[0].z - samplePoints[1].z))
                 / ((samplePoints[0].x - samplePoints[1].x)*(samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].x - samplePoints[2].x)*(samplePoints[0].z - samplePoints[1].z));
    plane.push_back(a);
    plane.push_back(-1);
    float c = a*(samplePoints[1].x - samplePoints[0].x) / (samplePoints[0].z - samplePoints[1].z);
    plane.push_back(c);
    plane.push_back(samplePoints[0].y - a*samplePoints[0].x - c*samplePoints[0].z);
    return plane;
}

  

float calculateError(vector<Point3f>& inliners, vector<float>& model)
{
    float tmp;
    for(size_t i = 0; i <inliners.size(); i++)
    {
        float tmp2 = distancePointPlane(inliners[i], model);
        tmp += tmp2 * tmp2;
    }
    tmp = sqrt(tmp);
    return tmp;
}

vector<Point3f> fisherYatesShuffle(vector<Point3f> data)
{
    vector<Point3f> tmp = data;
    for(int i=0; i < 3; i++)
    {
        int j = rand() % (tmp.size() - i);
        swap(tmp[tmp.size() - 1 - i], data[j]);
    }
    vector<Point3f> result;
    result.push_back(tmp[tmp.size() - 1]);
    result.push_back(tmp[tmp.size() - 2]);
    result.push_back(tmp[tmp.size() - 3]);
    return result;
}

vector<float> ransac(vector<Point3f>& data, int& k, int t, int d)
{  
    vector<float> bestfit;
    float besterror = 1000000;
    while(bestfit.size() == 0)
    {
        for(int iterations = 0; iterations < k; iterations++)
        {
            vector<Point3f> maybeinliners = fisherYatesShuffle(data);
            vector<float> maybemodel = getPlaneParams(maybeinliners);
            //FileStorage fr("Rezultat", FileStorage::APPEND);
            //fr << "radnom" << maybemodel;
            vector<Point3f> alsoinliners;
            for(size_t i = 0; i < data.size() ; i++)
            {
                //printf("%f", distancePointPlane(data[i], maybemodel));
                if(distancePointPlane(data[i], maybemodel) < t)
                {
                    alsoinliners.push_back(data[i]);
                }
            }
            //printf("%d", alsoinliners.size());
            if(alsoinliners.size() > d)
            {
                //printf("Pop sere");
                float thiserror = calculateError(alsoinliners, maybemodel);
                if(thiserror < besterror)
                {
                    bestfit = maybemodel;
                    besterror = thiserror;
                }
            }
        }
    }
    return bestfit;
}



//Multiplying of point and matrix______________________________________________________________________________________________________________________________________________________________________

cv::Point2f operator*(cv::Mat M, const cv::Point2f& p)
{
    cv::Mat_<double> src(3/*rows*/,1 /* cols */);

    src(0,0)=p.x;
    src(1,0)=p.y;
    src(2,0)=1.0;

    cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA
    return cv::Point2f(dst(0,0),dst(1,0));
}

//Convert From Homogenous_________________________________________________________________________________________________________________________________________________________________________________
vector<Point3f> convertFromHomogenous(Mat points4d)
{
    vector<Point3f> points3d;
    Point3f tmp;
    for(size_t i = 0; i < points4d.cols; i++)
    {
        tmp.x = points4d.at<float>(0,i) / points4d.at<float>(3,i);
        tmp.y = points4d.at<float>(1,i) / points4d.at<float>(3,i);
        tmp.z = points4d.at<float>(2,i) / points4d.at<float>(3,i);
        points3d.push_back(tmp);
    }
    return points3d;
}

//Main____________________________________________________________________________________________________________________________________________________________________________________________

int main(int argc, char** argv)
{


//Initialization________________________________________________________________________________________________________________________________________________________________________________
    Mat loadimg1 = imread("Images/sveska1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat loadimg2 = imread("Images/sveska2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img1, img2;

    FileStorage fr("calibration_data.yml", FileStorage::READ);

    Mat cameraMatrix;
    Mat distortionCoeff;

    fr["cameraMatrix"] >> cameraMatrix;
    fr["distortionCoeff"] >> distortionCoeff;

    undistort(loadimg1, img1, cameraMatrix, distortionCoeff);
    undistort(loadimg2, img2, cameraMatrix, distortionCoeff);



//Feature Matching______________________________________________________________________________________________________________________________________________________________________________

    // detecting keypoints

    Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(20,true), 0, 1000, 1000));
    //Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
    vector<KeyPoint> keypoints1;
    detector->detect(img1, keypoints1);

    vector<KeyPoint> keypoints2;
    detector->detect(img2, keypoints2);



    // computing descriptors
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("FREAK");
    Mat descriptors1;
    extractor->compute(img1, keypoints1, descriptors1);



    Mat descriptors2;
    extractor->compute(img2, keypoints2, descriptors2);


    // matching descriptors
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    BFMatcher matcher(NORM_HAMMING, false);
    vector<DMatch>  matches;
    matcher.match(descriptors1, descriptors2, matches);


//Triangulation____________________________________________________________________________________________________________________________________________________________________________

    vector<Point2f> matchingPoints1;
    vector<Point2f> matchingPoints2;
    for(int i=0; i<matches.size(); i++)
    {
        matchingPoints1.push_back(keypoints1[matches[i].queryIdx].pt);
        matchingPoints2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    vector<uchar> status;
    Mat fundamentalMatrix = findFundamentalMat(matchingPoints1, matchingPoints2, FM_RANSAC, 1, 0.99, status);
    fundamentalMatrix.convertTo(fundamentalMatrix, CV_64FC1);
    cameraMatrix.convertTo(cameraMatrix, CV_64FC1);
    distortionCoeff.convertTo(distortionCoeff, CV_64FC1);

    FileStorage fw("Provera.yml", FileStorage::WRITE);
    fw << "fundamental" << fundamentalMatrix;
    fw << "camera" << cameraMatrix;
    fw << "distortion" << distortionCoeff;
    fw << "camera transposed" << cameraMatrix.t();


    vector<Point2f> correctMatchingPoints1;
    vector<Point2f> correctMatchingPoints2;

    Mat_<double> tmp1(3/*rows*/,1 /* cols */);
    Mat_<double> tmp2(3/*rows*/,1 /* cols */);
    Mat_<double> formula;
    double error = 0;
    for(int i = 0; i < matches.size(); i++)
    {
        if(status[i] != 0)
        {
            correctMatchingPoints1.push_back(matchingPoints1[i]);
            tmp1(0,0)=matchingPoints1[i].x;
            tmp1(1,0)=matchingPoints1[i].y;
            tmp1(2,0)=1.0;
            correctMatchingPoints2.push_back(matchingPoints2[i]);
            tmp2(0,0)=matchingPoints2[i].x;
            tmp2(1,0)=matchingPoints2[i].y;
            tmp2(2,0)=1.0;
            formula = tmp2.t() * fundamentalMatrix * tmp1;
            error += formula(0,0);
        }
    }


    drawEpipolarLines(fundamentalMatrix, img1, img2, correctMatchingPoints1, correctMatchingPoints2, -1);




    error = error / correctMatchingPoints1.size();

    printf("%f \n",error);


    printf("%d \n", correctMatchingPoints1.size());

    vector<KeyPoint> keypointsc1;
    vector<KeyPoint> keypointsc2;
    for( size_t i = 0; i < correctMatchingPoints1.size(); i++ )
    {
        keypointsc1.push_back(KeyPoint(correctMatchingPoints1[i], 1.f));
        keypointsc2.push_back(KeyPoint(correctMatchingPoints2[i], 1.f));
    }

    vector<DMatch> correctMatches;
    for(size_t i = 0; i < matches.size(); i++)
    {
        if(status[i] != 0)
        {
            correctMatches.push_back(matches[i]);
        }
    }


    Mat essentialMatrix = cameraMatrix.t() * fundamentalMatrix * cameraMatrix;

    SVD svd(essentialMatrix, SVD::MODIFY_A);


    fw << "rand" << svd.u;

    Matx33d W((0, -1, 0), (1, 0, 0), (0, 0, 1));
    Matx33d Wt((0, 1, 0), (-1, 0, 0), (0, 0, 1));


    Mat rotationMatrix1 = svd.u * Mat(W) * svd.vt;;
    Mat translationMatrix1 = svd.u.col(2);
    translationMatrix1.convertTo(translationMatrix1, CV_64FC1);
    rotationMatrix1.convertTo(rotationMatrix1, CV_64FC1);
    Mat rotationMatrix2 = svd.u * Mat(Wt) * svd.vt;
    Mat translationMatrix2 = - svd.u.col(2);

    fw << "translation" << translationMatrix1;
    fw << "rotation" << rotationMatrix1;

    Mat R1, R2, P1, P2, Q;
    R1.convertTo(R1, CV_64FC1);
    R2.convertTo(R2, CV_64FC1);
    P1.convertTo(P1, CV_64FC1);
    P1.convertTo(P2, CV_64FC1);
    Q.convertTo(Q, CV_64FC1);


    stereoRectify(cameraMatrix, distortionCoeff, cameraMatrix , distortionCoeff, Size(9,6), rotationMatrix1, translationMatrix1, R1, R2, P1, P2, Q);
    //printf("Pop sere");

    Mat points4d(4,matches.size(),CV_64FC1);

    triangulatePoints(P1, P2, correctMatchingPoints1, correctMatchingPoints2, points4d);

    vector<Point3f> points3d = convertFromHomogenous(points4d);

    int k = (int)(log(1 - 0.99) / (log(1 - 0.65))) + 1;

    vector<float> plane = ransac(points3d, k, 1, 80);


    FileStorage store("Results.txt", FileStorage::WRITE);
    store << "Plane" << plane;
    store << "Points" << points3d;




//_________________________________________________________________________________________________________________________________________________________________________________________

    /*sort(matches.begin(), matches.end());

    vector<Point2f> sample1;
    vector<Point2f> sample2;
    for(int i=0;i<4;i++)
    {
        sample1.push_back(keypoints1[matches[i].queryIdx].pt);
        sample2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    FileStorage fs("fajldesc",FileStorage::WRITE);

    Mat perspectiveTransformMatrix;
    perspectiveTransformMatrix = getPerspectiveTransform(sample1, sample2);

    fs << "perspectiveTransformMatrix" << perspectiveTransformMatrix;
    fs << "rotationMatrix" << M;

//Success Rate______________________________________________________________________________________________________________________________________________________________________________

    int correct = 0;
    int all = 0;

    for(size_t i = 0; i < matches.size(); i++)
    {
        Point2f point1 = keypoints1[matches[i].queryIdx].pt;
        Point2f point2 = keypoints2[matches[i].trainIdx].pt;

        all++;

        Point2f tmp;
        vector<Point2f> p1;
        p1.push_back(point1);
        vector<Point2f> p2;
        p2.push_back(tmp);

        transform(p1, p2, M);

        Point2f t = p2[0] - point2;
        if(abs(t.x)<10  && abs(t.y)<10)
        {
            correct++;
        }
    }

    printf("%d", all);
    printf("\n");
    printf("%d", correct);
    */

//Results_________________________________________________________________________________________________________________________________________________________________________________________

    // drawing the results
    namedWindow("keypoints1", 1);
    namedWindow("keypoints2", 1);
    namedWindow("matches", 1);
    Mat img_kp1;
    Mat img_kp2;
    Mat img_matches;

    drawKeypoints(img1, keypointsc1, img_kp1);
    drawKeypoints(img2, keypointsc2, img_kp2);

    vector<DMatch> bestmatches;
    vector<KeyPoint> bestkeypoints1;
    vector<KeyPoint> bestkeypoints2;

    for(int i=0; i<50;i++)
    {
        bestmatches.push_back(matches[i]);
        KeyPoint point1 = keypoints1[matches[i].queryIdx];
        bestkeypoints1.push_back(point1);
        KeyPoint point2 = keypoints2[matches[i].trainIdx];
        bestkeypoints2.push_back(point2);
    }

    drawMatches(img1, keypoints1, img2, keypoints2, correctMatches, img_matches);
    imshow("keypoints1", img_kp1);
    imshow("keypoints2", img_kp2);
    imshow("matches", img_matches);
    waitKey(0);

    waitKey();

    return 0;
}
