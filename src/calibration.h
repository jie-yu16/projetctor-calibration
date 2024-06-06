#pragma once
#include "stdafx.h"

void compute_save_result(vector<vector<Point3f>> object_points, vector<vector<Point2f>> cam_points_seq,
	Mat cameraMatrix1, Mat distCoeffs1, vector<Mat>rvecsMat, vector<Mat>tvecsMat, string file_name, int image_num, Size board_size, int flag);

void circle_blob_test(); // blob detecting method for calibration

