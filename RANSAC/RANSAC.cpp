#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <random>
#include <time.h>

using namespace std;
using namespace cv;

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

vector<float> ransac(vector<Point3f> &data, int &k, int t, int d)
{
    vector<float> bestfit;
    float besterror = 1000000;
    while (bestfit.size() == 0)
    {
        for (int iterations = 0; iterations < k; iterations++)
        {
            vector<Point3f> maybeinliners = fisherYatesShuffle(data);
            vector<float> maybemodel = getPlaneParams(maybeinliners);
            //FileStorage fr("Rezultat", FileStorage::APPEND);
            //fr << "radnom" << maybemodel;
            vector<Point3f> alsoinliners;
            for (size_t i = 0; i < data.size(); i++)
            {
                printf("%f", distancePointPlane(data[i], maybemodel));
                if (distancePointPlane(data[i], maybemodel) < t)
                {
                    alsoinliners.push_back(data[i]);
                }
            }
            printf("%d", alsoinliners.size());
            if (alsoinliners.size() > d)
            {
                printf("Pop sere");
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

int main(int argc, char **argv)
{
    srand(time(NULL));
    vector<Point3f> data;
    data.push_back(Point3f(0, 0, 0));
    data.push_back(Point3f(2, 1, 0));
    data.push_back(Point3f(4, 2, 0));
    data.push_back(Point3f(6, 3, 0));
    data.push_back(Point3f(8, 4, 0));
    data.push_back(Point3f(10, 5, 0));
    data.push_back(Point3f(12, 6, 0));
    data.push_back(Point3f(14, 7, 0));
    data.push_back(Point3f(2, 5, 4));
    data.push_back(Point3f(15, 1, 6));

    int k = (int)(log(1 - 0.99) / (log(1 - 0.64))) + 1;

    //printf("%d", k);

    vector<float> plane = ransac(data, k, 1, 3);

    FileStorage fw("Rezultat", FileStorage::WRITE);
    fw << "Plane" << plane;
    return 0;
}
