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

#define PI 3.14159265

void drawPoints(Mat &img, vector<Point2f> points)
{
	for (size_t i = 0; i < points.size(); i++)
	{
		circle(img, points[i], 1, CV_RGB(100, 200, 100), 2, 8, 0);
	}
}

float distancePointLine(const Point2f point, const Vec<float, 3> &line)
{
	//Line is given as a*x + b*y + c = 0
	return fabsf(line(0) * point.x + line(1) * point.y + line(2)) / std::sqrt(line(0) * line(0) + line(1) * line(1));
}

void cv::computeCorrespondEpilines(InputArray _points, int whichImage, InputArray _Fmat, OutputArray _lines)
{
	Mat points = _points.getMat(), F = _Fmat.getMat();
	int npoints = points.checkVector(2);
	if (npoints < 0)
		npoints = points.checkVector(3);
	CV_Assert(npoints >= 0 && (points.depth() == CV_32F || points.depth() == CV_32S));

	_lines.create(npoints, 1, CV_32FC3, -1, true);
	CvMat c_points = points, c_lines = _lines.getMat(), c_F = F;
	cvComputeCorrespondEpilines(&c_points, whichImage, &c_F, &c_lines);
}

void drawEpipolarLines(cv::Mat &F, cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, float inlierDistance = -1)
{

	CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
	cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
	cv::Rect rect1(0, 0, img1.cols, img1.rows);
	cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

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
	std::vector<cv::Vec<float, 3>> epilines1, epilines2;
	cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
	cv::computeCorrespondEpilines(points2, 2, F, epilines2);

	CV_Assert(points1.size() == points2.size() &&
			  points2.size() == epilines1.size() &&
			  epilines1.size() == epilines2.size());

	cv::RNG rng(0);
	for (size_t i = 0; i < points1.size(); i++)
	{
		if (inlierDistance > 0)
		{
			if (distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
				distancePointLine(points2[i], epilines1[i]) > inlierDistance)
			{
				continue;
			}
		}
		cv::Scalar color(rng(256), rng(256), rng(256));

		Mat tmp2 = outImg(rect2);
		Mat tmp1 = outImg(rect1);

		cv::line(tmp2,
				 cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
				 cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
				 color);
		cv::circle(tmp1, points1[i], 3, color, -1, CV_AA);

		cv::line(tmp1,
				 cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
				 cv::Point(img2.cols, -(epilines2[i][2] + epilines2[i][0] * img2.cols) / epilines2[i][1]),
				 color);
		cv::circle(tmp2, points2[i], 3, color, -1, CV_AA);
	}
	cv::imshow("Epl", outImg);
	cv::waitKey(1);
}

float distancePointPlane(const Point3f &point, const vector<float> &plane)
{
	return fabsf(plane[0] * point.x + plane[1] * point.y + plane[2] * point.z + plane[3]) / sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
}

vector<float> getPlaneParams(vector<Point3f> &samplePoints)
{
	vector<float> plane;
	float a = ((samplePoints[0].y - samplePoints[1].y) * (samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].y - samplePoints[2].y) * (samplePoints[0].z - samplePoints[1].z)) / ((samplePoints[0].x - samplePoints[1].x) * (samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].x - samplePoints[2].x) * (samplePoints[0].z - samplePoints[1].z));
	plane.push_back(a);
	plane.push_back(-1);
	float c = a * (samplePoints[1].x - samplePoints[0].x) / (samplePoints[0].z - samplePoints[1].z);
	plane.push_back(c);
	plane.push_back(samplePoints[0].y - a * samplePoints[0].x - c * samplePoints[0].z);
	return plane;
}

float calculateError(vector<Point3f> &inliners, vector<float> &model)
{
	float tmp;
	for (size_t i = 0; i < inliners.size(); i++)
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
	for (int i = 0; i < 3; i++)
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

vector<float> ransac(vector<Point3f> &data, int &k, float t, float d)
{
	vector<float> bestfit;
	float besterror = 1000000;
	cout << "a";
	while (bestfit.size() == 0)
	{
		for (int iterations = 0; iterations < k; iterations++)
		{
			vector<Point3f> maybeinliners = fisherYatesShuffle(data);
			vector<float> maybemodel = getPlaneParams(maybeinliners);
			vector<Point3f> alsoinliners;
			for (size_t i = 0; i < data.size(); i++)
			{
				//cout << distancePointPlane(data[i], maybemodel) << "\n";
				if (distancePointPlane(data[i], maybemodel) < t)
				{
					//cout << distancePointPlane(data[i], maybemodel) << "\n";
					alsoinliners.push_back(data[i]);
				}
			}
			if (alsoinliners.size() > d)
			{
				float thiserror = calculateError(alsoinliners, maybemodel);
				if (thiserror < besterror)
				{
					bestfit = maybemodel;
					besterror = thiserror;
				}
			}
		}
	}
	return bestfit;
}

vector<Point3f> convertFromHomogenous(Mat points4d)
{
	points4d.convertTo(points4d, CV_32F);
	vector<Point3f> points3d;
	Point3f tmp;
	for (size_t i = 0; i < points4d.cols; i++)
	{
		tmp.x = points4d.at<float>(0, i) / points4d.at<float>(3, i);
		tmp.y = points4d.at<float>(1, i) / points4d.at<float>(3, i);
		tmp.z = points4d.at<float>(2, i) / points4d.at<float>(3, i);
		points3d.push_back(tmp);
	}
	return points3d;
}

bool DecomposeEtoRandT(const Mat_<float> E, Mat_<float> &R1, Mat_<float> &R2, Mat_<float> &t1, Mat_<float> &t2)
{
	SVD svd(E, SVD::MODIFY_A);

	float singular_values_ratio = fabsf(svd.w.at<float>(0) / svd.w.at<float>(1));
	if (singular_values_ratio > 1)
		singular_values_ratio = 1 / singular_values_ratio;
	if (singular_values_ratio < 0.7)
	{
		cout << "singular values of essential matrix are too far apart\n";
		return false;
	}

	Matx33f W(0, -1, 0,
			  1, 0, 0,
			  0, 0, 1);
	Matx33f Wt(0, 1, 0,
			   -1, 0, 0,
			   0, 0, 1);
	R1 = svd.u * Mat(W) * svd.vt;
	R2 = svd.u * Mat(Wt) * svd.vt;
	t1 = svd.u.col(2);
	t2 = -svd.u.col(2);
	return true;
}

template <typename T, typename V, typename K, typename L, typename I>
void keepVectorsByStatus(vector<T> &f1, vector<V> &f2, vector<K> &f3, vector<L> &f4, vector<I>& f5, const vector<uchar> &status)
{
	vector<T> oldf1 = f1;
	vector<V> oldf2 = f2;
	vector<K> oldf3 = f3;
	vector<L> oldf4 = f4;
	vector<I> oldf5 = f5;
	f1.clear();
	f2.clear();
	f3.clear();
	f4.clear();
	f5.clear();
	for (int i = 0; i < status.size(); ++i)
	{
		if (status[i])
		{
			f1.push_back(oldf1[i]);
			f2.push_back(oldf2[i]);
			f3.push_back(oldf3[i]);
			f4.push_back(oldf4[i]);
			f5.push_back(oldf5[i]);
		}
	}
}

vector<float> transformPlane(Mat R, Mat T, vector<float> plane, bool inverse)
{
	R.convertTo(R, CV_32F);
	T.convertTo(T, CV_32F);
	Mat Rp(3, 3, CV_32F);
	if (inverse)
	{
		Rp = R.inv();
	}
	else
	{
		Rp = R;
	}
	Mat n(3, 1, CV_32F);
	Mat N(3, 1, CV_32F);
	n.at<float>(0, 0) = plane[0];
	n.at<float>(1, 0) = plane[1];
	n.at<float>(2, 0) = plane[2];
	N = Rp * n;
	vector<float> newPlane;
	newPlane.push_back(N.at<float>(0, 0));
	newPlane.push_back(N.at<float>(1, 0));
	newPlane.push_back(N.at<float>(2, 0));
	float d = plane[3] - T.at<float>(0, 0) * plane[0] - T.at<float>(1, 0) * plane[1] - T.at<float>(2, 0) * plane[2];
	newPlane.push_back(d);
	return newPlane;
}

vector<Point3f> transformPoints(Mat R, Mat T, vector<Point3f> &points, bool inverse)
{
	vector<Point3f> outpoints;
	R.convertTo(R, CV_32F);
	T.convertTo(T, CV_32F);
	Mat Rp(3, 3, CV_32F);
	Mat Tp(3, 1, CV_32F);
	if (inverse)
	{
		Rp = R.inv();
		Tp = -T;
	}
	else
	{
		Rp = R;
		Tp = T;
	}
	Mat tmp(3, 1, CV_32F);
	Mat res1(3, 1, CV_32F);
	for (int i = 0; i < points.size(); i++)
	{
		tmp.at<float>(0, 0) = points[i].x;
		tmp.at<float>(1, 0) = points[i].y;
		tmp.at<float>(2, 0) = points[i].z;

		res1 = Rp * tmp;
		res1.at<float>(0, 0) += Tp.at<float>(0, 0);
		res1.at<float>(1, 0) += Tp.at<float>(1, 0);
		res1.at<float>(2, 0) += Tp.at<float>(2, 0);

		Point3f tmpp;
		tmpp.x = res1.at<float>(0, 0);
		tmpp.y = res1.at<float>(1, 0);
		tmpp.z = res1.at<float>(2, 0);
		outpoints.push_back(tmpp);
	}
	return outpoints;
}

bool triangulateAndCheckReproj(const Mat R, const Mat t, vector<Point2f> &correctMatchingPoints1, vector<Point2f> &correctMatchingPoints2,
							   Mat cameraMatrix, vector<Point3f> &points3dfirst, vector<Point3f> &points3dsecond)
{
	FileStorage fw("Core_triangulate.yml", FileStorage::WRITE);
	Mat_<float> P = Mat::eye(3, 4, CV_32FC1);
	Mat_<float> P2 = (Mat_<float>(3, 4) << R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2), t.at<float>(0), //2,1
					  R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2), t.at<float>(1),						//0,2
					  R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2), t.at<float>(2));						//1,0

	Mat normalizedTrackedPts, normalizedBootstrapPts;
	undistortPoints(Mat(correctMatchingPoints2), normalizedTrackedPts, cameraMatrix, Mat());
	undistortPoints(Mat(correctMatchingPoints1), normalizedBootstrapPts, cameraMatrix, Mat());

	//triangulate
	Mat pt_3d_h(4, correctMatchingPoints2.size(), CV_32FC1);
	triangulatePoints(P, P2, normalizedBootstrapPts, normalizedTrackedPts, pt_3d_h);
	Mat pt_3d;
	convertPointsFromHomogeneous(Mat(pt_3d_h.t()).reshape(4, 1), pt_3d);
	vector<Point3f> first, second;
	first = convertFromHomogenous(pt_3d_h);
	second = transformPoints(R, t, first, true);
	fw << "first" << first;
	fw << "second" << second;

	vector<uchar> status1(first.size(), 0);
	vector<uchar> status2(second.size(), 0);
	for (int i = 0; i < first.size(); i++)
	{
		status1[i] = (first[i].z > 0) ? 1 : 0;
		status2[i] = (second[i].z > 0) ? 1 : 0;
	}
	int count1 = countNonZero(status1);
	int count2 = countNonZero(status2);
	float percentage1 = ((float)count1 / (float)first.size());
	float percentage2 = ((float)count2 / (float)second.size());
	cout << count1 << "/" << first.size() << " = " << percentage1 * 100.0 << "% are in front of first camera \t";
	cout << count2 << "/" << second.size() << " = " << percentage2 * 100.0 << "% are in front of second camera \t";
	if (percentage1 < 0.75 || percentage2 < 0.75)
	{
		printf("\n");
		points3dfirst.clear();
		points3dsecond.clear();
		return false; //less than 75% of the points are in front of the camera
	}
	else
	{
		vector<Point2f> remainingPoints1, remainingPoints2;
		for (int i = 0; i < first.size(); i++)
		{
			if (status1[i] != 0 && status2[i] != 0)
			{
				points3dfirst.push_back(first[i]);
				points3dsecond.push_back(second[i]);
				remainingPoints1.push_back(correctMatchingPoints1[i]);
				remainingPoints2.push_back(correctMatchingPoints2[i]);
			}
		}
		correctMatchingPoints1.clear();
		correctMatchingPoints2.clear();
		correctMatchingPoints1 = remainingPoints1;
		correctMatchingPoints2 = remainingPoints2;
		//calculate reprojection
		Vec3f rvec(0, 0, 0); //Rodrigues(R ,rvec);
		Vec3f tvec(0, 0, 0); // = P.col(3);
		fw << "rvec" << rvec;
		fw << "tvec" << tvec;
		vector<Point2f> reprojected_pt_set1;
		projectPoints(points3dfirst, rvec, tvec, cameraMatrix, Mat(), reprojected_pt_set1);
		fw << "reprojected" << reprojected_pt_set1;
		float reprojErr;
		for (int i = 0; i < reprojected_pt_set1.size(); i++)
		{
			reprojErr += sqrtf((reprojected_pt_set1[i].x - correctMatchingPoints1[i].x) * (reprojected_pt_set1[i].x - correctMatchingPoints1[i].x) +
							   (reprojected_pt_set1[i].y - correctMatchingPoints1[i].y) * (reprojected_pt_set1[i].y - correctMatchingPoints1[i].y));
		}
		reprojErr = reprojErr / reprojected_pt_set1.size();
		cout << "reprojection Error " << reprojErr << "\n";
		if (reprojErr < 5)
		{
			cout << "\nFiltering Points ... \n";  
			vector<uchar> status(correctMatchingPoints1.size(), 0);
			for (int i = 0; i < correctMatchingPoints1.size(); ++i)
			{
				float distance = sqrtf((reprojected_pt_set1[i].x - correctMatchingPoints1[i].x) * (reprojected_pt_set1[i].x - correctMatchingPoints1[i].x) +
									   (reprojected_pt_set1[i].y - correctMatchingPoints1[i].y) * (reprojected_pt_set1[i].y - correctMatchingPoints1[i].y));
				status[i] = (distance < 3);
			}
			keepVectorsByStatus(correctMatchingPoints1, correctMatchingPoints2, reprojected_pt_set1, points3dfirst, points3dsecond, status);
			cout << "keeping " << correctMatchingPoints1.size() << " nicely reprojected points \n";
			fw << "Points" << points3dfirst;
			reprojErr = 0;
			for (int i = 0; i < reprojected_pt_set1.size(); i++)
			{
				float distance = sqrtf((reprojected_pt_set1[i].x - correctMatchingPoints1[i].x) * (reprojected_pt_set1[i].x - correctMatchingPoints1[i].x) +
								   (reprojected_pt_set1[i].y - correctMatchingPoints1[i].y) * (reprojected_pt_set1[i].y - correctMatchingPoints1[i].y));
				reprojErr += distance;
			}
			reprojErr = reprojErr / reprojected_pt_set1.size();
			cout << "New reprojection error "<< reprojErr << "\n";
			return true;
		}
	}
	return false;
}

void sintetickiPodaci(Mat cameraMatrix, float t, Mat_<float> wp, vector<Point2f> &points1, vector<Point2f> &points2) // t - translacija kamere
{
	Mat_<float> tmp;
	Mat_<float> res;
	Mat_<float> wpn = wp.clone();
	for (int i = 0; i < wp.rows; i++)
	{
		tmp = wp.row(i).t();
		res = cameraMatrix * tmp;
		points1.push_back(Point2f(res.at<float>(0, 0) / res.at<float>(2, 0), res.at<float>(1, 0) / res.at<float>(2, 0)));
	}
	for (int i = 0; i < wpn.rows; i++)
	{
		wpn.at<float>(i, 0) -= t;
	}
	for (int i = 0; i < wpn.rows; i++)
	{
		tmp = wpn.row(i).t();
		res = cameraMatrix * tmp;
		points2.push_back(Point2f(res.at<float>(0, 0) / res.at<float>(2, 0), res.at<float>(1, 0) / res.at<float>(2, 0)));
	}
}

float checkResults(Mat img, Mat cameraMatrix, vector<Point3f> points3dfirst, vector<Point2f> correctMatchingPoints1)
{
	Vec3f rvec(0, 0, 0);
	Vec3f tvec(0, 0, 0);
	vector<Point2f> reprojected;
	projectPoints(points3dfirst, rvec, tvec, cameraMatrix, Mat(), reprojected);
	float error = 0;
	for (int i = 0; i < reprojected.size(); i++)
	{
		error += sqrtf((reprojected[i].x - correctMatchingPoints1[i].x) * (reprojected[i].x - correctMatchingPoints1[i].x) +
					   (reprojected[i].y - correctMatchingPoints1[i].y) * (reprojected[i].y - correctMatchingPoints1[i].y));
	}
	error = error / reprojected.size();
	Mat img_found, img_reprojected;
	vector<KeyPoint> keypointsFound, keypointsReprojected;
	for (size_t i = 0; i < correctMatchingPoints1.size(); i++)
	{
		keypointsFound.push_back(KeyPoint(correctMatchingPoints1[i], 1.f));
		keypointsReprojected.push_back(KeyPoint(reprojected[i], 1.f));
	}
	drawKeypoints(img, keypointsFound, img_found);
	drawKeypoints(img, keypointsReprojected, img_reprojected);
	Mat outimg(img.rows, img.cols * 2, CV_8UC3);
	Rect rect1(0, 0, img.cols, img.rows);
	Rect rect2(img.cols, 0, img.cols, img.rows);
	img_found.copyTo(outimg(rect1));
	img_reprojected.copyTo(outimg(rect2));
	imshow("Check Results", outimg);
	waitKey(0);
	return error;
}

float findFeaturePoints(Mat img1, Mat img2, vector<Point2f> &correctMatchingPoints1, vector<Point2f> &correctMatchingPoints2, Mat &fundamentalMatrix)
{
	FileStorage fw("Core_FeaturePoints.yml", FileStorage::WRITE);

	vector<KeyPoint> correctKeyPoints1;
	vector<KeyPoint> correctKeyPoints2;
	vector<DMatch> correctMatches;

	//Feature Matching______________________________________________________________________________________________________________________________________________________________________________

	// detecting keypoints
	cout << "\nDetecting feature points ... \n";
	Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(20, true), 0, 1000, 1000));
	//Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");

	vector<KeyPoint> keypoints1;
	detector->detect(img1, keypoints1);

	vector<KeyPoint> keypoints2;
	detector->detect(img2, keypoints2);

	// computing descriptors
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("BRISK");
	Mat descriptors1;
	extractor->compute(img1, keypoints1, descriptors1);

	Mat descriptors2;
	extractor->compute(img2, keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher(NORM_HAMMING, false);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	vector<Point2f> matchingPoints1;
	vector<Point2f> matchingPoints2;

	for (int i = 0; i < matches.size(); i++)
	{
		matchingPoints1.push_back(keypoints1[matches[i].queryIdx].pt);
		matchingPoints2.push_back(keypoints2[matches[i].trainIdx].pt);
	}
	Size winSize = Size(10, 10);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
	cornerSubPix(img1, matchingPoints1, winSize, zeroZone, criteria);
	cornerSubPix(img1, matchingPoints1, winSize, zeroZone, criteria);

	cout << "Filtering feature points ... \t";
	vector<uchar> status;
	fundamentalMatrix = findFundamentalMat(matchingPoints1, matchingPoints2, FM_RANSAC, 1, 0.99, status);
	fundamentalMatrix.convertTo(fundamentalMatrix, CV_32F);
	Mat_<float> tmp1(3, 1);
	Mat_<float> tmp2(3, 1);
	Mat_<float> formula;
	float error = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		if (status[i] != 0)
		{
			correctMatchingPoints1.push_back(matchingPoints1[i]);
			tmp1(0, 0) = matchingPoints1[i].x;
			tmp1(1, 0) = matchingPoints1[i].y;
			tmp1(2, 0) = 1.0;
			correctMatchingPoints2.push_back(matchingPoints2[i]);
			tmp2(0, 0) = matchingPoints2[i].x;
			tmp2(1, 0) = matchingPoints2[i].y;
			tmp2(2, 0) = 1.0;
			formula = tmp2.t() * fundamentalMatrix * tmp1;
			error += formula(0, 0);
		}
	}
	error = error / correctMatchingPoints1.size();
	cout << "Left " << correctMatchingPoints1.size() << " out of " << matchingPoints1.size() << " points \t";
	vector<KeyPoint> keypointsc1;
	vector<KeyPoint> keypointsc2;
	for (size_t i = 0; i < correctMatchingPoints1.size(); i++)
	{
		keypointsc1.push_back(KeyPoint(correctMatchingPoints1[i], 1.f));
		keypointsc2.push_back(KeyPoint(correctMatchingPoints2[i], 1.f));
	}

	for (size_t i = 0; i < matches.size(); i++)
	{
		if (status[i] != 0)
		{
			correctMatches.push_back(matches[i]);
		}
	}
	return error;
}

void showFeaturePoints(Mat img1, Mat img2, vector<Point2f> correctMatchingPoints1, vector<Point2f> correctMatchingPoints2, Mat fundamentalMatrix)
{
	namedWindow("FeaturePoints", CV_WINDOW_FULLSCREEN);
	vector<KeyPoint> keypoints1, keypoints2;
	for (size_t i = 0; i < correctMatchingPoints1.size(); i++)
	{
		keypoints1.push_back(KeyPoint(correctMatchingPoints1[i], 1.f));
		keypoints2.push_back(KeyPoint(correctMatchingPoints2[i], 1.f));
	}
	Mat img_kp1, img_kp2;
	drawKeypoints(img1, keypoints1, img_kp1);
	drawKeypoints(img2, keypoints2, img_kp2);
	Mat outimg(img1.rows, img1.cols * 2, CV_8UC3);
	Rect rect1(0, 0, img1.cols, img1.rows);
	Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	img_kp1.copyTo(outimg(rect1));
	img_kp2.copyTo(outimg(rect2));
	imshow("FeaturePoints", outimg);
	drawEpipolarLines(fundamentalMatrix, img1, img2, correctMatchingPoints1, correctMatchingPoints2, -1);
	waitKey(0);
}

float getAngle(vector<float> chessboard, vector<float> plane)
{
	float normChessBoard = sqrtf(chessboard[0] * chessboard[0] + chessboard[1] * chessboard[1] + chessboard[2] * chessboard[2]);
    float normPlane = sqrtf(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    float angle = plane[0] * chessboard[0] + plane[1] * chessboard[1] + plane[2] * chessboard[2];
    angle = angle / (normPlane * normChessBoard);
    angle = acos(angle) * 180.0 / PI;
	return angle;
}

vector<float> getPlane(Mat cameraMatrixread, Mat distortionCoeffread, Mat fundamentalMatrix, vector<Point2f> &correctMatchingPoints1, vector<Point2f> &correctMatchingPoints2, vector<Point3f> &points3dfirst, vector<Point3f> &points3dsecond)
{
	FileStorage fw("Core_getPlane.yml", FileStorage::WRITE);

	vector<float> plane;
	/*Mat_<float> cameraMatrix = (Mat_<float>(3,3) <<
							4,0,2,
							0,4,2,
							0,0,1);

	Mat_<float> wp = (Mat_<float>(16  ,3) <<
							1000, 500, 11,
							1100, 500, 10,
							1200, 500, 10,
							1300, 500, 10,
							1000, 600, 10,
							1100, 600, 10,
							1200, 600, 10,
							1300, 600, 10,
							1000, 700, 11,
							1100, 700, 10,
							1200, 700, 10,
							1300, 700, 10,
							1000, 800, 10,
							1100, 800, 10,
							1200, 800, 10,
							1300, 800, 10);

	sintetickiPodaci(cameraMatrix, -300, wp, correctMatchingPoints1, correctMatchingPoints2);
	Mat first(1000, 1000, CV_8UC3, Scalar(0, 0, 0));
	Mat second(1000, 1000, CV_8UC3, Scalar(0, 0, 0));
	drawPoints(first, correctMatchingPoints1);
	drawPoints(second, correctMatchingPoints2);
	imwrite("first.jpg", first);
	imwrite("second.jpg", second);

	*/

	Mat cameraMatrix(3, 3, CV_32F);
	Mat distortionCoeff(5, 1, CV_32F);

	cameraMatrixread.convertTo(cameraMatrix, CV_32F);
	distortionCoeffread.convertTo(distortionCoeff, CV_32F);

	fw << "Fundamental" << fundamentalMatrix;
	fw << "cameraMatrix" << cameraMatrix;
	Mat_<float> E = cameraMatrix.t() * fundamentalMatrix * cameraMatrix;
	fw << "Essential" << E;

	Mat_<float> R1(3, 3);
	Mat_<float> R2(3, 3);
	Mat_<float> t1(3, 3);
	Mat_<float> t2(3, 3);
	fw << "R1" << R1;

	DecomposeEtoRandT(E, R1, R2, t1, t2);

	if (determinant(R1) + 1.0 < 1e-09)
	{
		E = -E;
		DecomposeEtoRandT(E, R1, R2, t1, t2);
	}
	cout << "\nError of Rotation in stereo pair " << fabsf(determinant(R2)) - 1.0 << "\n";
	if ((fabsf(determinant(R2)) - 1.0 > 1e-06))
	{
		cout << "det(R) != +-1.0 \n";
		return plane;
	}

	printf("Traingulating 1st time \t");
	if (!triangulateAndCheckReproj(R1, t1, correctMatchingPoints1, correctMatchingPoints2, cameraMatrix, points3dfirst, points3dsecond))
	{
		printf("Traingulating 2nd time \t");
		if (!triangulateAndCheckReproj(R1, t2, correctMatchingPoints1, correctMatchingPoints2, cameraMatrix, points3dfirst, points3dsecond))
		{
			printf("Traingulating 3rd time \t");
			if (!triangulateAndCheckReproj(R2, t2, correctMatchingPoints1, correctMatchingPoints2, cameraMatrix, points3dfirst, points3dsecond))
			{
				printf("Traingulating 4th time \t");
				if (!triangulateAndCheckReproj(R2, t1, correctMatchingPoints1, correctMatchingPoints2, cameraMatrix, points3dfirst, points3dsecond))
				{
					printf("Triangulation failed \n");
				}
			}
		}
	}
	//printf("Dosao");

	int datasetSize = correctMatchingPoints1.size();
	//int k = (int)(log(1 - 0.99) / (log(1 - 0.1))) + 1;
	int k = 100;
	cout << "\nFitting plane ... \n";
	plane = ransac(points3dfirst, k, 1, 3 * datasetSize / 4);
	cout << "Plane fitted successfully \n";
	return plane;
}
