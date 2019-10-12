
bool SimpleAdHocTracker::triangulateAndCheckReproj(const Mat &P, const Mat &P1)
{
    //undistort
    Mat normalizedTrackedPts, normalizedBootstrapPts;
    undistortPoints(Points<float>(trackedFeatures), normalizedTrackedPts, camMat, Mat());
    undistortPoints(Points<float>(bootstrap_kp), normalizedBootstrapPts, camMat, Mat());

    //triangulate
    Mat pt_3d_h(4, trackedFeatures.size(), CV_32FC1);
    cv::triangulatePoints(P, P1, normalizedBootstrapPts, normalizedTrackedPts, pt_3d_h);
    Mat pt_3d;
    convertPointsFromHomogeneous(Mat(pt_3d_h.t()).reshape(4, 1), pt_3d);
    //    cout << pt_3d.size() << endl;
    //    cout << pt_3d.rowRange(0,10) << endl;

    vector<uchar> status(pt_3d.rows, 0);
    for (int i = 0; i < pt_3d.rows; i++)
    {
        status[i] = (pt_3d.at<Point3f>(i).z > 0) ? 1 : 0;
    }
    int count = countNonZero(status);

    double percentage = ((double)count / (double)pt_3d.rows);
    cout << count << "/" << pt_3d.rows << " = " << percentage * 100.0 << "% are in front of camera";
    if (percentage < 0.75)
        return false; //less than 75% of the points are in front of the camera

    //calculate reprojection
    cv::Mat_<double> R = P(cv::Rect(0, 0, 3, 3));
    Vec3d rvec(0, 0, 0); //Rodrigues(R ,rvec);
    Vec3d tvec(0, 0, 0); // = P.col(3);
    vector<Point2f> reprojected_pt_set1;
    projectPoints(pt_3d, rvec, tvec, camMat, Mat(), reprojected_pt_set1);
    //    cout << Mat(reprojected_pt_set1).rowRange(0,10) << endl;
    vector<Point2f> bootstrapPts_v = Points<float>(bootstrap_kp);
    Mat bootstrapPts = Mat(bootstrapPts_v);
    //    cout << bootstrapPts.rowRange(0,10) << endl;

    double reprojErr = cv::norm(Mat(reprojected_pt_set1), bootstrapPts, NORM_L2) / (double)bootstrapPts_v.size();
    cout << "reprojection Error " << reprojErr;
    if (reprojErr < 5)
    {
        vector<uchar> status(bootstrapPts_v.size(), 0);
        for (int i = 0; i < bootstrapPts_v.size(); ++i)
        {
            status[i] = (norm(bootstrapPts_v[i] - reprojected_pt_set1[i]) < 20.0);
        }

        trackedFeatures3D.clear();
        trackedFeatures3D.resize(pt_3d.rows);
        pt_3d.copyTo(Mat(trackedFeatures3D));

        keepVectorsByStatus(trackedFeatures, trackedFeatures3D, status);
        cout << "keeping " << trackedFeatures.size() << " nicely reprojected points";
        bootstrapping = false;
        return true;
    }
    return false;
}
