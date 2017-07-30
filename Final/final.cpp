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

vector<float> getPlaneParams(vector<Point3f>& pon)
{
		vector<float> plane;
		/*
		float a = ((samplePoints[0].y - samplePoints[1].y)*(samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].y - samplePoints[2].y)*(samplePoints[0].z - samplePoints[1].z))
								 / ((samplePoints[0].x - samplePoints[1].x)*(samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].x - samplePoints[2].x)*(samplePoints[0].z - samplePoints[1].z));
		plane.push_back(a);
		plane.push_back(-1);
		float c = a*(samplePoints[1].x - samplePoints[0].x) / (samplePoints[0].z - samplePoints[1].z);
		plane.push_back(c);
		plane.push_back(samplePoints[0].y - a*samplePoints[0].x - c*samplePoints[0].z);
		*/
		float a = (pon[1].y - pon[0].y) * (pon[2].z - pon[0].z) - (pon[2].y - pon[0].y) * (pon[1].z - pon[0].z);
		float b = (pon[1].z - pon[0].z) * (pon[2].x - pon[0].x) - (pon[2].z - pon[0].z) * (pon[1].x - pon[0].x);
		float c = (pon[1].x - pon[0].x) * (pon[2].y - pon[0].y) - (pon[2].x - pon[0].x) * (pon[1].y - pon[0].y);
		float d = -(a*pon[0].x + b*pon[0].y + c*pon[0].z);
		plane.push_back(a);
		plane.push_back(b);
		plane.push_back(c);
		plane.push_back(d);
		printf("%f %f %f %f \n", a,b,c,d);
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
		cv::Mat_<double> src(3,1);

		src(0,0)=p.x;
		src(1,0)=p.y;
		src(2,0)=1.0;

		cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA
		return cv::Point2f(dst(0,0),dst(1,0));
}

vector<float> transformPlane(Mat R ,Mat T, vector<float> plane)
{
		Mat_<float> rotation(4,4);
		Mat Rp;
		transpose(R, Rp);
		for(int i=0; i < 3; i++)
		{
				for(int j=0; j < 3; j++)
				{
						rotation.at<float>(i,j) = Rp.at<float>(i,j);
				}
		}
		rotation.at<float>(0,3) = 0.0;
		rotation.at<float>(1,3) = 0.0;
		rotation.at<float>(2,3) = 0.0;
		rotation.at<float>(3,0) = 0.0;
		rotation.at<float>(3,1) = 0.0;
		rotation.at<float>(3,2) = 0.0;
		rotation.at<float>(3,3) = 1.0;
		Mat_<float> translation(4,4);
		for(int i=0; i < 4; i++)
		{
				for(int j=0; j < 4; j++)
				{
						if(i==j)
						{
								translation.at<float>(i,j) = 1.0;
						}
						else
						{
								translation.at<float>(i,j) = 0.0;
						}
				}
		}
		translation.at<float>(0,3) = -T.at<float>(0,0);
		translation.at<float>(1,3) = -T.at<float>(1,0);
		translation.at<float>(2,3) = -T.at<float>(2,0);

		Mat_<float> p1(4,1);
		p1.at<float>(0,0)=1.0;
		p1.at<float>(1,0)=1.0;
		p1.at<float>(2,0)=(-plane[3] - plane[0] - plane[1])/plane[2];
		p1.at<float>(3,0)=1.0;
		Mat_<float> p1pp = rotation*p1;
		Mat_<float> p1p = translation * p1pp;
		Point3f p1t = Point3f(p1p.at<float>(0,0),p1p.at<float>(1,0),p1p.at<float>(2,0));
		Mat_<float> p2(4,1);
		p2.at<float>(0,0)= -1.0;
		p2.at<float>(1,0)= 1.0;
		p2.at<float>(2,0)=(-plane[3] + plane[0] - plane[1])/plane[2];
		p2.at<float>(3,0)= 1.0;
		Mat_<float> p2pp = rotation*p2;
		Mat_<float> p2p = translation * p2pp;
		Point3f p2t = Point3f(p2p.at<float>(0,0),p2p.at<float>(1,0),p2p.at<float>(2,0));
		Mat_<float> p3(4,1);
		p3.at<float>(0,0)= 1.0;
		p3.at<float>(1,0)= -1.0;
		p3.at<float>(2,0)=(-plane[3] - plane[0] + plane[1])/plane[2];
		p3.at<float>(3,0)=1.0;
		Mat_<float> p3pp = rotation*p3;
		Mat_<float> p3p = translation * p3pp;
		Point3f p3t = Point3f(p3p.at<float>(0,0),p3p.at<float>(1,0),p3p.at<float>(2,0));
		vector<Point3f> newPoints;
		newPoints.push_back(p1t);
		newPoints.push_back(p2t);
		newPoints.push_back(p3t);

		vector<float> newPlane = getPlaneParams(newPoints);

		                    FileStorage store("test.txt", FileStorage::WRITE);
		                    store << "Points" << newPoints;
		                    store << "Plane" << newPlane;
							store << "Rotation" << rotation;
							store << "Rotation small" << Rp;
							store << "Translation" << translation;
							store << "P1" << p1t;
							store << "P2" << p2t;
							store << "P3" << p3t;
		                    store.release();

		return newPlane;
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

//Feature Matching, Filter,Triangulation, Ransac_________________________________________________________________________________________________________________________________________________________________________

vector<float> getPlane(Mat img1, Mat img2, Mat cameraMatrix, Mat distortionCoeff, vector<Point2f>& correctMatchingPoints1, vector<Point2f>& correctMatchingPoints2,
														vector<KeyPoint>& correctKeyPoints1, vector<KeyPoint>& correctKeyPoints2,vector<DMatch>& correctMatches, vector<Point3f>& points3d )
{
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

		printf("\n %d \n", keypoints1.size());

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


		Mat_<double> tmp1(3,1);
		Mat_<double> tmp2(3,1);
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

		//drawEpipolarLines(fundamentalMatrix, img1, img2, correctMatchingPoints1, correctMatchingPoints2, -1);




		error = error / correctMatchingPoints1.size();

		printf("%f \n",error);


		printf("%d \n", correctMatchingPoints1.size());

		printf("Pop sere");

		for( size_t i = 0; i < correctMatchingPoints1.size(); i++ )
		{
				correctKeyPoints1.push_back(KeyPoint(correctMatchingPoints1[i], 1.f));
				correctKeyPoints2.push_back(KeyPoint(correctMatchingPoints2[i], 1.f));
		}

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

		Mat points4d;

		triangulatePoints(P1, P2, correctMatchingPoints1, correctMatchingPoints2, points4d);

		points3d = convertFromHomogenous(points4d);

		int datasetSize = correctMatchingPoints1.size();
		int k = (int)(log(1 - 0.99) / (log(1 - 0.64))) + 1;

		vector<float> plane = ransac(points3d, k, 1, datasetSize/2);


		FileStorage store("Results.txt", FileStorage::WRITE);
		store << "Plane" << plane;
		store << "Points" << points3d;


	 return plane;
}

//Drawing of points_____________________________________________________________________________________________________________________________________________________________________________________________
void drawPoints(Mat& img, vector<Point2f> points)
{
		for(size_t i = 0; i < points.size(); i++)
		{
				circle(img, points[i], 5, CV_RGB(100,200,100), 3, 8, 0);
		}
}


//Main____________________________________________________________________________________________________________________________________________________________________________________________

int mdodosdaain(int argc, char** argv)
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

		vector<Point2f> correctMatchingPoints1, correctMatchingPoints2;
		vector<KeyPoint> correctKeyPoints1, correctKeyPoints2;
		vector<DMatch> correctMatches;
		vector<Point3f> points3d;

		//vector<float> plane = getPlane(img1, img2, cameraMatrix, distortionCoeff, correctMatchingPoints1, correctMatchingPoints2, correctKeyPoints1, correctKeyPoints2, correctMatches, points3d);

		VideoCapture vid(1);;
		if(!vid.isOpened())
		{
				return 0;
		}
		namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
		int framesPerSecond = 60;
		Mat frame;
		Mat undistortFrame, finalFrame;
		Mat currentFrame, undistortCurrentFrame;
		Mat oldFrame;
		Mat initMat1, initMat2;
		vector<Point2f> prevPts, nextPts;
		vector<uchar> status;
		vector<float> err;
		int init = 0;
		TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
		Size winSize(31,31);
		vector<float> plane;
		Mat tvector, rvector;
		vector<uchar> inliers;

		bool ok = true;
		while(ok)
		{
				if(!vid.read(frame))
				{
						break;
				}
				undistort(frame, undistortFrame, cameraMatrix, distortionCoeff);
				cvtColor(frame, finalFrame, CV_RGB2GRAY);
				imshow("Webcam", frame);
				char character = waitKey(100 / framesPerSecond);
				switch(character)
				{
						case ' ':
								if(init == 0)
								{
										initMat1 = finalFrame.clone();
								}
								if(init == 1)
								{
										initMat2 = finalFrame.clone();
										oldFrame = initMat2.clone();
										prevPts = correctMatchingPoints2;
								}
								if(init == 2)
								{
										currentFrame = finalFrame.clone();
								}
								init++;
								break;
						case 'i':
								plane = getPlane(initMat1, initMat2, cameraMatrix, distortionCoeff, correctMatchingPoints1, correctMatchingPoints2, correctKeyPoints1, correctKeyPoints2, correctMatches, points3d);
								prevPts = correctMatchingPoints2;
								ok = false;
								break;
						case 's':
								if(plane.size() != 0)
								{
										//currentFrame = finalFrame.clone();
										//undistort(currentFrame, undistortCurrentFrame, cameraMatrix, distortionCoeff);
										//calcOpticalFlowPyrLK(oldFrame, undistortCurrentFrame, prevPts, nextPts, status, err, winSize, 3, termcrit, 0, 0.001);
										//solvePnPRansac(points3d, nextPts, cameraMatrix, distortionCoeff, rvector, tvector, true, 6, 5, nextPts.size()/2, inliers, CV_P3P);
								}
								break;
						case 27:
								return 0;
								break;
				}
		}

		undistort(currentFrame, undistortCurrentFrame, cameraMatrix, distortionCoeff);
		calcOpticalFlowPyrLK(oldFrame, undistortCurrentFrame, prevPts, nextPts, status, err, winSize, 3, termcrit, 0, 0.001);
		printf("\n %d \n %d", points3d.size(), nextPts.size());
		solvePnPRansac(points3d, nextPts, cameraMatrix, distortionCoeff, rvector, tvector, true, 6, 5.0, nextPts.size()/2, inliers, CV_P3P);

		FileStorage fw("Vectori", FileStorage::WRITE);
		Mat rMat;
		Rodrigues(rvector, rMat);
		fw << "rMat" << rMat;
		fw << "tvector" << tvector;


		namedWindow("keypoints1", CV_WINDOW_AUTOSIZE);
		namedWindow("keypoints2", CV_WINDOW_AUTOSIZE);
		namedWindow("OF points", CV_WINDOW_AUTOSIZE);
		imshow("keypoints1", initMat1);
		imshow("keypoints2", initMat2);

		/*calcOpticalFlowPyrLK(initMat1, initMat2, correctMatchingPoints1, nextPts, status, err, winSize, 3, termcrit, 0, 0.001);
		drawPoints(initMat1, correctMatchingPoints1);
		Mat tmp = initMat2.clone();
		drawPoints(initMat2, correctMatchingPoints2);
		drawPoints(tmp, nextPts);
		vector<float> plane = getPlane(initMat1, initMat2, cameraMatrix, distortionCoeff, correctMatchingPoints1, correctMatchingPoints2, correctKeyPoints1, correctKeyPoints2, correctMatches, points3d);
		*/
		imshow("keypoints1", initMat1);
		imshow("keypoints2", initMat2);
		//imshow("OF points", tmp);



//Results_________________________________________________________________________________________________________________________________________________________________________________________

		// drawing the results
		namedWindow("matches", 1);

		//drawKeypoints(img1, correctKeyPoints1, img_kp1);
		//drawKeypoints(img2, correctKeyPoints2, img_kp2);



		///drawMatches(img1, correctKeyPoints1, img2, correctKeyPoints2, correctMatches, img_matches);
		printf("pop sere");
		//imshow("keypoints1", img_kp1);
		//imshow("keypoints2", img_kp2);
		//imshow("matches", img_matches);
		waitKey(0);
		return 0;
}
