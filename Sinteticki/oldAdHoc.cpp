bool triangulateAndCheckReproj(const Mat &P, const Mat &P1,
                               vector<Point2f> correctMatchingPoints1, vector<Point2f> correctMatchingPoints2,
                               Mat cameraMatrix, vector<Point3f> &points3d)
{
    vector<Point2f> trackedFeatures = correctMatchingPoints1;
    vector<Point2f> bootstrap_kp = correctMatchingPoints2;

    //undistort
    Mat normalizedTrackedPts, normalizedBootstrapPts;
    undistortPoints(Mat(trackedFeatures), normalizedTrackedPts, cameraMatrix, Mat());
    undistortPoints(Mat(bootstrap_kp), normalizedBootstrapPts, cameraMatrix, Mat());
    //triangulate
    Mat pt_3d_h(4, trackedFeatures.size(), CV_32F);
    triangulatePoints(P, P1, normalizedTrackedPts, normalizedBootstrapPts, pt_3d_h);
    //Mat pt_3d; convertPointsFromHomogeneous(Mat(pt_3d_h.t()).reshape(4, 1),pt_3d);
    vector<Point3f> points3dp;
    points3dp = convertFromHomogenous(pt_3d_h);
    //    cout << pt_3d.size() << endl;
    //    cout << pt_3d.rowRange(0,10) << endl;
    vector<uchar> status(points3dp.size(), 0);
    for (int i = 0; i < points3dp.size(); i++)
    {
        status[i] = (points3dp[i].z > 0) ? 1 : 0;
    }
    int count = countNonZero(status);

    float percentage = ((float)count / (float)points3dp.size());
    if (percentage < 0.9)
    {
        return false;
    }
    else
    {
        points3d.clear();
        points3d = convertFromHomogenous(pt_3d_h);
        return true;
    }
}
