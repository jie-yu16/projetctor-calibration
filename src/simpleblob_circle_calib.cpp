#include "calibration.h"

// #define DEBUG_BLOB_DETECTOR
void compute_save_result(vector<vector<Point3f>> object_points, vector<vector<Point2f>> cam_points_seq,
	Mat cameraMatrix1, Mat distCoeffs1, vector<Mat>rvecsMat, vector<Mat>tvecsMat, string file_name, int image_num, Size board_size, int flag)
{
	ofstream fout(file_name);
	vector<int> point_counts;  // 每幅图像中角点的数量，假定每幅图均能看到全部角点  
	for (int i = 0; i < image_num; i++)
	{
		point_counts.push_back(board_size.width * board_size.height);
	}
	string name1, name2;
	if (flag == 0)
	{
		name1 = "./outputs/compute_reproject_cam_corner1.txt";
		name2 = "./outputs/compute_reproject_cam_corner2.txt";
	}
	else
	{
		name1 = "./outputs/compute_reproject_pro_corner1.txt";
		name2 = "./outputs/compute_reproject_pro_corner2.txt";
	}
	ofstream fout1(name1);
	ofstream fout2(name2);

	cout << "标定完成！\n";
	//对标定结果进行评价  
	cout << "开始评价标定结果………………\n";
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
	cout << "\t每幅图像的标定误差：\n";
	fout << "每幅图像的标定误差：\n";
	for (int i = 0; i < image_num; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix1, distCoeffs1, image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = cam_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);

			fout1 << image_points2[j].x << ' ' << image_points2[j].y << endl;
			fout2 << tempImagePoint[j].x << ' ' << tempImagePoint[j].y << endl;

		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= sqrt(point_counts[i]);
		std::cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	std::cout << "总体平均误差：" << total_err / image_num << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_num << "像素" << endl << endl;
	std::cout << "评价完成！" << endl;
	//保存定标结果      
	std::cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fout << "内参数矩阵：" << endl;
	fout << cameraMatrix1 << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs1 << endl << endl << endl;
	for (int i = 0; i < image_num; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rvecsMat[i] << endl;
		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(rvecsMat[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << tvecsMat[i] << endl << endl;
	}

	fout.close();
}

struct Center
{
	Point2d location;
	double confidence;
	double radius;
};

struct Params
{
	double thresholdStep;
	double minThreshold;
	double maxThreshold;

	double minRepeatability;
	double minDistBetweenBlobs;
	bool filterByColor;
	double blobColor;
	bool filterByArea;
	double minArea;
	double maxArea;

	bool filterByCircularity;
	double minCircularity;
	double maxCircularity;

	bool filterByInertia;
	double minInertiaRatio;
	double maxInertiaRatio;
	bool filterByConvexity;
	double minConvexity;
	double maxConvexity;

	//Params() :thresholdStep(), minThreshold(), maxThreshold(), minRepeatability(), minDistBetweenBlobs(), filterByColor(), minArea(),
	//	maxArea(), filterByCircularity(), minCircularity(), maxCircularity(), filterByInertia(), minInertiaRatio(), maxInertiaRatio(), filterByConvexity(), minConvexity(), maxConvexity(){}
};


class SimpleBlob1
{
public:
	Params params;
	void initialize();
	void findBlobs(const cv::Mat& image, const cv::Mat& binaryImage, vector<Center> &centers, bool **D);
	void findBlobs_pro(const cv::Mat& image, const cv::Mat& binaryImage, vector<Center>& centers, Point2f** pro_image, bool **D, Mat H_pro);
	void detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, bool **D);	
	void detectImpl_pro(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, Point2f **pro_image, bool **D, Mat H_pro);
};

void SimpleBlob1::initialize()
{
	params.thresholdStep = 10;    //二值化的阈值步长，即公式1的t
	params.minThreshold = 100;   //二值化的起始阈值，即公式1的T1
	params.maxThreshold = 210;    //二值化的终止阈值，即公式1的T2
	//重复的最小次数，只有属于灰度图像斑点的那些二值图像斑点数量大于该值时，该灰度图像斑点才被认为是特征点
	params.minRepeatability = 2;
	//最小的斑点距离，不同二值图像的斑点间距离小于该值时，被认为是同一个位置的斑点，否则是不同位置上的斑点
	params.minDistBetweenBlobs = 10;

	params.filterByColor = true;    //斑点颜色的限制变量
	params.blobColor = 0;    //表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点

	params.filterByArea = true;    //斑点面积的限制变量
	params.minArea = 50;    //斑点的最小面积
	params.maxArea = 2000;    //斑点的最大面积

	params.filterByCircularity = false;    //斑点圆度的限制变量，默认是不限制
	params.minCircularity = 0.6f;    //斑点的最小圆度
	//斑点的最大圆度，所能表示的float类型的最大值
	params.maxCircularity = std::numeric_limits<float>::max();

	params.filterByInertia = true;    //斑点惯性率的限制变量
	//minInertiaRatio = 0.6;
	params.minInertiaRatio = 0.1f;    //斑点的最小惯性率
	params.maxInertiaRatio = std::numeric_limits<float>::max();    //斑点的最大惯性率

	params.filterByConvexity = true;    //斑点凸度的限制变量
	//minConvexity = 0.8;
	params.minConvexity = 0.8f;    //斑点的最小凸度
	params.maxConvexity = std::numeric_limits<float>::max();    //斑点的最大凸度
}

//image为输入的灰度图像
//binaryImage为二值图像
//centers表示该二值图像的斑点
void SimpleBlob1::findBlobs(const cv::Mat& image, const cv::Mat& binaryImage, vector<Center> & centers, bool **D)
{
	(void)image;
	centers.clear();    //斑点变量清零

	vector < vector<Point> > contours;    //定义二值图像的斑点的边界像素变量
	Mat tmpBinaryImage = binaryImage.clone();    //复制二值图像
	//调用findContours函数，找到当前二值图像的所有斑点的边界
	findContours(tmpBinaryImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
	  Mat keypointsImage;
	  cvtColor( binaryImage, keypointsImage, CV_GRAY2RGB );
	  //Mat contoursImage;
	  //cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
	  //namedWindow("contours", CV_WINDOW_NORMAL);
	  //drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
	  //imshow("contours", contoursImage );
	  //waitKey();
#endif
	//遍历当前二值图像的所有斑点
	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		//结构类型Center代表着斑点，它包括斑点的中心位置，半径和权值

		bool flag = 0;
		for (int i = 0; i < contours[contourIdx].size(); i++)
		{
			if (D[contours[contourIdx][i].y][contours[contourIdx][i].x] == 0)
			{
				flag = 1;
				break;
			}
		}
		if (flag == 1)
		{
			continue;
		}
		Center center;    //斑点变量
		//初始化斑点中心的置信度，也就是该斑点的权值
		center.confidence = 1;
		//调用moments函数，得到当前斑点的矩
		Moments moms = moments(Mat(contours[contourIdx]));
		if (params.filterByArea)    //斑点面积的限制
		{
			double area = moms.m00;    //零阶矩即为二值图像的面积
			//如果面积超出了设定的范围，则不再考虑该斑点
			if (area < params.minArea || area >= params.maxArea)
				continue;
		}

		if (params.filterByCircularity)    //斑点圆度的限制
		{
			double area = moms.m00;    //得到斑点的面积
			//计算斑点的周长
			double perimeter = arcLength(Mat(contours[contourIdx]), true);
			//由公式3得到斑点的圆度
			double ratio = 4 * CV_PI * area / (perimeter * perimeter);
			//如果圆度超出了设定的范围，则不再考虑该斑点
			if (ratio < params.minCircularity || ratio >= params.maxCircularity)
				continue;
		}

		if (params.filterByInertia)    //斑点惯性率的限制
		{
			//计算公式13中最右侧等式中的开根号的值
			double denominator = sqrt(pow(2 * moms.mu11, 2) + pow(moms.mu20 - moms.mu02, 2));
			const double eps = 1e-2;    //定义一个极小值
			double ratio;
			if (denominator > eps)
			{
				//cosmin和sinmin用于计算图像协方差矩阵中较小的那个特征值λ2
				double cosmin = (moms.mu20 - moms.mu02) / denominator;
				double sinmin = 2 * moms.mu11 / denominator;
				//cosmin和sinmin用于计算图像协方差矩阵中较大的那个特征值λ1
				double cosmax = -cosmin;
				double sinmax = -sinmin;
				//imin为λ2乘以零阶中心矩μ00
				double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
				//imax为λ1乘以零阶中心矩μ00
				double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
				ratio = imin / imax;    //得到斑点的惯性率
			}
			else
			{
				ratio = 1;    //直接设置为1，即为圆
			}
			//如果惯性率超出了设定的范围，则不再考虑该斑点
			if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
				continue;
			//斑点中心的权值定义为惯性率的平方
			center.confidence = ratio * ratio;
		}

		if (params.filterByConvexity)    //斑点凸度的限制
		{
			vector < Point > hull;    //定义凸壳变量
			//调用convexHull函数，得到该斑点的凸壳
			convexHull(Mat(contours[contourIdx]), hull);
			//分别得到斑点和凸壳的面积，contourArea函数本质上也是求图像的零阶矩
			double area = contourArea(Mat(contours[contourIdx]));
			double hullArea = contourArea(Mat(hull));
			double ratio = area / hullArea;    //公式5，计算斑点的凸度
			//如果凸度超出了设定的范围，则不再考虑该斑点
			if (ratio < params.minConvexity || ratio >= params.maxConvexity)
				continue;
		}

		//根据公式7，计算斑点的质心
		center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

		if (params.filterByColor)    //斑点颜色的限制
		{
			//判断一下斑点的颜色是否正确
			if (binaryImage.at<uchar>(cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
				continue;
		}

		//compute blob radius
		{
			vector<double> dists;    //定义距离队列
			//遍历该斑点边界上的所有像素
			for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
			{
				Point2d pt = contours[contourIdx][pointIdx];    //得到边界像素坐标
				//计算该点坐标与斑点中心的距离，并放入距离队列中
				dists.push_back(norm(center.location - pt));
			}
			std::sort(dists.begin(), dists.end());    //距离队列排序
			//计算斑点的半径，它等于距离队列中中间两个距离的平均值
			center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
		}

		centers.push_back(center);    //把center变量压入centers队列中


#ifdef DEBUG_BLOB_DETECTOR
		cout << center.location << endl;
		cout << center.radius << endl;
		circle( keypointsImage, center.location, 1, Scalar(0,0,255), 1 );
#endif
	}

	//imshow("bk", binaryImage);
	//cout << centers.size() << endl;
	//waitKey();
#ifdef DEBUG_BLOB_DETECTOR
	imshow("bk", keypointsImage );
	waitKey();
#endif
}
void SimpleBlob1::detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, bool **D)
{
	//TODO: support mask
	keypoints.clear();    //特征点变量清零
	Mat grayscaleImage;
	//把彩色图像转换为二值图像
	if (image.channels() == 3)
		cvtColor(image, grayscaleImage, CV_BGR2GRAY);
	else
		grayscaleImage = image;
	//二维数组centers表示所有得到的斑点，第一维数据表示的是灰度图像斑点，第二维数据表示的是属于该灰度图像斑点的所有二值图像斑点 
	//结构类型Center代表着斑点，它包括斑点的中心位置，半径和权值
	vector < vector<Center> > centers;
	//遍历所有阈值，进行二值化处理
	for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
	{
		Mat binarizedImage;
		//调用threshold函数，把灰度图像grayscaleImage转换为二值图像binarizedImage
		threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);

#ifdef DEBUG_BLOB_DETECTOR
		    //Mat keypointsImage;
		    //cvtColor( binarizedImage, keypointsImage, CV_GRAY2RGB );
		namedWindow("binary_image", CV_WINDOW_NORMAL);
		imshow("binary_image", binarizedImage);
		waitKey();

#endif
		//变量curCenters表示该二值图像内的所有斑点
		vector < Center > curCenters;
		//调用findBlobs函数，对二值图像binarizedImage检测斑点，得到curCenters
		findBlobs(grayscaleImage, binarizedImage, curCenters, D);
		//newCenters表示在当前二值图像内检测到的不属于已有灰度图像斑点的那些二值图像斑点
		vector < vector<Center> > newCenters;

		//// show the detected blob images
		//cout << "curCenters:" << curCenters.size() << endl;
		//namedWindow("keypoints", CV_WINDOW_NORMAL);
		//Mat outImg;
		//if (outImg.channels() == 1)
		//	cvtColor(image, outImg, CV_GRAY2RGB);

		//for (size_t i = 0; i < curCenters.size(); i++)
		//{
		//	circle(outImg, curCenters[i].location, curCenters[i].radius, Scalar(255, 0, 255), -1);
		//}
		////drawKeypoints(image, keypoints, outImg);
		//imshow("keypoints", outImg);
		//waitKey();

		//遍历该二值图像内的所有斑点
		for (size_t i = 0; i < curCenters.size(); i++)
		{
#ifdef DEBUG_BLOB_DETECTOR
			//      circle(keypointsImage, curCenters[i].location, curCenters[i].radius, Scalar(0,0,255),-1);
#endif
			// isNew表示的是当前二值图像斑点是否为新出现的斑点
			bool isNew = true;
			//遍历已有的所有灰度图像斑点，判断该二值图像斑点是否为新的灰度图像斑点

			for (size_t j = 0; j < centers.size(); j++)
			{
				//属于该灰度图像斑点的中间位置的那个二值图像斑点代表该灰度图像斑点，并把它的中心坐标与当前二值图像斑点的中心坐标比较，计算它们的距离dist
				double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);
				//如果距离大于所设的阈值，并且距离都大于上述那两个二值图像斑点的半径，则表示该二值图像的斑点是新的斑点
				isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][centers[j].size() / 2].radius && dist >= curCenters[i].radius;
				//如果不是新的斑点，则需要把它添加到属于它的当前（即第j个）灰度图像的斑点中，因为通过上面的距离比较可知，当前二值图像斑点属于当前灰度图像斑点
				if (!isNew)
				{
					//把当前二值图像斑点存入当前（即第j个）灰度图像斑点数组的最后位置
					centers[j].push_back(curCenters[i]);
					//得到构成该灰度图像斑点的所有二值图像斑点的数量
					size_t k = centers[j].size() - 1;
					//按照半径由小至大的顺序，把新得到的当前二值图像斑点放入当前灰度图像斑点数组的适当位置处，由于二值化阈值是按照从小到大的顺序设置，所以二值图像斑点本身就是按照面积的大小顺序被检测到的，因此此处的排序处理要相对简单一些
					while (k > 0 && centers[j][k].radius < centers[j][k - 1].radius)
					{
						centers[j][k] = centers[j][k - 1];
						k--;
					}
					centers[j][k] = curCenters[i];
					//由于当前二值图像斑点已经找到了属于它的灰度图像斑点，因此退出for循环，无需再遍历已有的灰度图像斑点
					break;
				}
			}
			if (isNew)    //当前二值图像斑点是新的斑点
			{
				//把当前斑点存入newCenters数组内
				newCenters.push_back(vector<Center>(1, curCenters[i]));
				//centers.push_back(vector<Center> (1, curCenters[i]));
			}
		}
		//把当前二值图像内的所有newCenters复制到centers内

		std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));

		//cout << " center_size:"<< centers.size() << endl;
#ifdef DEBUG_BLOB_DETECTOR
		//    imshow("binarized", keypointsImage );
		//waitKey();
#endif
	}    //所有二值图像斑点检测完毕
	//遍历所有灰度图像斑点，得到特征点信息
	cout << "Cam_Centers:" << centers.size() << endl;
	for (size_t i = 0; i < centers.size(); i++)
	{
		//如果属于当前灰度图像斑点的二值图像斑点的数量小于所设阈值，则该灰度图像斑点不是特征点
		if (centers[i].size() < params.minRepeatability)
			continue;
		Point2d sumPoint(0, 0);
		double normalizer = 0;
		//遍历属于当前灰度图像斑点的所有二值图像斑点
		for (size_t j = 0; j < centers[i].size(); j++)
		{
			sumPoint += centers[i][j].confidence * centers[i][j].location;    //公式2的分子
			normalizer += centers[i][j].confidence;    //公式2的分母
		}
		sumPoint *= (1. / normalizer);    //公式2，得到特征点的坐标位置
		//保存该特征点的坐标位置和尺寸大小
		KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius));
		keypoints.push_back(kpt);    //保存该特征点
	}




#ifdef DEBUG_BLOB_DETECTOR
	namedWindow("keypoints", CV_WINDOW_NORMAL);
	Mat outImg = image.clone();
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		circle(outImg, keypoints[i].pt, keypoints[i].size, Scalar(255, 0, 255), -1);
	}
	//drawKeypoints(image, keypoints, outImg);
	imshow("keypoints", outImg);
	waitKey();
#endif
}

void SimpleBlob1::findBlobs_pro(const cv::Mat& image, const cv::Mat& binaryImage, vector<Center>& centers, Point2f** pro_image, bool **D, Mat H_pro)
{
	(void)image;
	centers.clear();    //斑点变量清零

	vector < vector<Point> > contours;    //定义二值图像的斑点的边界像素变量
	Mat tmpBinaryImage = binaryImage.clone();    //复制二值图像
	//调用findContours函数，找到当前二值图像的所有斑点的边界
	findContours(tmpBinaryImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
	Mat keypointsImage;
	cvtColor(binaryImage, keypointsImage, CV_GRAY2RGB);
	//Mat contoursImage;
	//cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
	//namedWindow("contours", CV_WINDOW_NORMAL);
	//drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
	//imshow("contours", contoursImage );
	//waitKey();
#endif
	//遍历当前二值图像的所有斑点
	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{

		//结构类型Center代表着斑点，它包括斑点的中心位置，半径和权值
		Center center;    //斑点变量
		//初始化斑点中心的置信度，也就是该斑点的权值
		center.confidence = 1;
		//调用moments函数，得到当前斑点的矩


		//decode for projector_image
		vector<Point2f> contours_pro;
		bool flag = 0;
		for (int pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
		{
			if (D[contours[contourIdx][pointIdx].y][contours[contourIdx][pointIdx].x] == 0) { flag = 1; }
			Point2f temp = pro_image[contours[contourIdx][pointIdx].y][contours[contourIdx][pointIdx].x];
			temp.y = temp.y / 2;
			contours_pro.push_back(temp);
		}
		if (flag == 1)
		{
			continue;
		}

		vector<Point2f> contours_H;
		for (int pointIdx = 0; pointIdx < contours_pro.size(); pointIdx++)
		{
			Mat temp = (Mat_<double>(3, 1) << (double)contours_pro[pointIdx].x, (double)contours_pro[pointIdx].y, 1);;
			Mat temp1 = H_pro.inv() * temp;
			Point2f d_temp_point(temp1.at<double>(0) / temp1.at<double>(2), temp1.at<double>(1) / temp1.at<double>(2));
			contours_H.push_back(d_temp_point);
		}
		Moments moms = moments(Mat(contours_H));

		//Moments moms = moments(Mat(contours_pro));
		if (params.filterByArea)    //斑点面积的限制
		{
			double area = moms.m00;    //零阶矩即为二值图像的面积
			//如果面积超出了设定的范围，则不再考虑该斑点
			if (area < params.minArea || area >= params.maxArea)
				continue;
		}

		if (params.filterByCircularity)    //斑点圆度的限制
		{
			double area = moms.m00;    //得到斑点的面积
			//计算斑点的周长
			double perimeter = arcLength(Mat(contours_pro), true);
			//由公式3得到斑点的圆度
			double ratio = 4 * CV_PI * area / (perimeter * perimeter);
			//如果圆度超出了设定的范围，则不再考虑该斑点
			if (ratio < params.minCircularity || ratio >= params.maxCircularity)
				continue;
		}

		if (params.filterByInertia)    //斑点惯性率的限制
		{
			//计算公式13中最右侧等式中的开根号的值
			double denominator = sqrt(pow(2 * moms.mu11, 2) + pow(moms.mu20 - moms.mu02, 2));
			const double eps = 1e-2;    //定义一个极小值
			double ratio;
			if (denominator > eps)
			{
				//cosmin和sinmin用于计算图像协方差矩阵中较小的那个特征值λ2
				double cosmin = (moms.mu20 - moms.mu02) / denominator;
				double sinmin = 2 * moms.mu11 / denominator;
				//cosmin和sinmin用于计算图像协方差矩阵中较大的那个特征值λ1
				double cosmax = -cosmin;
				double sinmax = -sinmin;
				//imin为λ2乘以零阶中心矩μ00
				double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
				//imax为λ1乘以零阶中心矩μ00
				double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
				ratio = imin / imax;    //得到斑点的惯性率
			}
			else
			{
				ratio = 1;    //直接设置为1，即为圆
			}
			//如果惯性率超出了设定的范围，则不再考虑该斑点
			if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
				continue;
			//斑点中心的权值定义为惯性率的平方
			center.confidence = ratio * ratio;
		}

		if (params.filterByConvexity)    //斑点凸度的限制
		{
			vector < Point > hull, hull_1;    //定义凸壳变量
			//调用convexHull函数，得到该斑点的凸壳
			convexHull(Mat(contours[contourIdx]), hull);

			//分别得到斑点和凸壳的面积，contourArea函数本质上也是求图像的零阶矩
			double area = contourArea(Mat(contours[contourIdx]));
			double hullArea = contourArea(Mat(hull));

			double ratio = area / hullArea;    //公式5，计算斑点的凸度
			//如果凸度超出了设定的范围，则不再考虑该斑点
			if (ratio < params.minConvexity || ratio >= params.maxConvexity)
				continue;
		}

		//vector<Point2f> contours_H;
		//for (int pointIdx = 0; pointIdx < contours_pro.size(); pointIdx++)
		//{
		//	Mat temp = (Mat_<double>(3, 1) << (double)contours_pro[pointIdx].x, (double)contours_pro[pointIdx].y, 1);;
		//	Mat temp1 = H_pro.inv() * temp;
		//	Point2f d_temp_point(temp1.at<double>(0) / temp1.at<double>(2), temp1.at<double>(1) / temp1.at<double>(2));
		//	contours_H.push_back(d_temp_point);
		//}
		Moments mu = moments(Mat(contours_H));
		Point2d detect_circle_H(mu.m10 / mu.m00, mu.m01 / mu.m00);
		Mat temp = (Mat_<double>(3, 1) << (double)detect_circle_H.x, (double)detect_circle_H.y, 1);
		Mat temp1 = H_pro * temp;

		//根据公式7，计算斑点的质心
		center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

		//cout << "no H:" << center.location << endl;

		center.location = Point2d(temp1.at<double>(0) / temp1.at<double>(2), temp1.at<double>(1) / temp1.at<double>(2));

		//cout<<"H:" << center.location << endl;



		//if (params.filterByColor)    //斑点颜色的限制
		//{
		//	//判断一下斑点的颜色是否正确
		//	if (binaryImage.at<uchar>(cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
		//		continue;
		//}

		//compute blob radius
		{
			vector<double> dists;    //定义距离队列
			//遍历该斑点边界上的所有像素
			for (size_t pointIdx = 0; pointIdx < contours_pro.size(); pointIdx++)
			{
				Point2d pt = contours_pro[pointIdx];    //得到边界像素坐标
				//计算该点坐标与斑点中心的距离，并放入距离队列中
				dists.push_back(norm(center.location - pt));
			}
			std::sort(dists.begin(), dists.end());    //距离队列排序
			//计算斑点的半径，它等于距离队列中中间两个距离的平均值
			center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
		}

		centers.push_back(center);    //把center变量压入centers队列中


#ifdef DEBUG_BLOB_DETECTOR
		cout << center.location << endl;
		cout << center.radius << endl;
		circle(keypointsImage, center.location, 1, Scalar(0, 0, 255), 1);
#endif
	}

	//imshow("bk", binaryImage);
	//cout << centers.size() << endl;
	//waitKey();
#ifdef DEBUG_BLOB_DETECTOR
	imshow("bk", keypointsImage);
	waitKey();
#endif
}
void SimpleBlob1::detectImpl_pro(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, Point2f** pro_image, bool **D, Mat H_pro)
{
	//TODO: support mask
	keypoints.clear();    //特征点变量清零
	Mat grayscaleImage;
	//把彩色图像转换为二值图像
	if (image.channels() == 3)
		cvtColor(image, grayscaleImage, CV_BGR2GRAY);
	else
		grayscaleImage = image;
	//二维数组centers表示所有得到的斑点，第一维数据表示的是灰度图像斑点，第二维数据表示的是属于该灰度图像斑点的所有二值图像斑点 
	//结构类型Center代表着斑点，它包括斑点的中心位置，半径和权值
	vector < vector<Center> > centers;
	//遍历所有阈值，进行二值化处理
	for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
	{
		Mat binarizedImage;
		//调用threshold函数，把灰度图像grayscaleImage转换为二值图像binarizedImage
		threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);

#ifdef DEBUG_BLOB_DETECTOR
		    //Mat keypointsImage;
		    //cvtColor( binarizedImage, keypointsImage, CV_GRAY2RGB );
		namedWindow("binary_image", CV_WINDOW_NORMAL);
		imshow("binary_image", binarizedImage);
		waitKey();

#endif
		//变量curCenters表示该二值图像内的所有斑点
		vector < Center > curCenters;
		//调用findBlobs函数，对二值图像binarizedImage检测斑点，得到curCenters
		findBlobs_pro(grayscaleImage, binarizedImage, curCenters, pro_image, D, H_pro);
		//newCenters表示在当前二值图像内检测到的不属于已有灰度图像斑点的那些二值图像斑点
		vector < vector<Center> > newCenters;

		//cout << "curCenters:" << curCenters.size() << endl;
		//namedWindow("keypoints", CV_WINDOW_NORMAL);
		//Mat outImg;
		//if (outImg.channels() == 1)
		//	cvtColor(image, outImg, CV_GRAY2RGB);

		//for (size_t i = 0; i < curCenters.size(); i++)
		//{
		//	circle(outImg, curCenters[i].location, curCenters[i].radius, Scalar(255, 0, 255), -1);
		//}
		////drawKeypoints(image, keypoints, outImg);
		//imshow("keypoints", outImg);
		//waitKey();



		//遍历该二值图像内的所有斑点
		for (size_t i = 0; i < curCenters.size(); i++)
		{
#ifdef DEBUG_BLOB_DETECTOR
			//      circle(keypointsImage, curCenters[i].location, curCenters[i].radius, Scalar(0,0,255),-1);
#endif
			// isNew表示的是当前二值图像斑点是否为新出现的斑点
			bool isNew = true;
			//遍历已有的所有灰度图像斑点，判断该二值图像斑点是否为新的灰度图像斑点

			for (size_t j = 0; j < centers.size(); j++)
			{
				//属于该灰度图像斑点的中间位置的那个二值图像斑点代表该灰度图像斑点，并把它的中心坐标与当前二值图像斑点的中心坐标比较，计算它们的距离dist
				double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);
				//如果距离大于所设的阈值，并且距离都大于上述那两个二值图像斑点的半径，则表示该二值图像的斑点是新的斑点
				isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][centers[j].size() / 2].radius && dist >= curCenters[i].radius;
				//如果不是新的斑点，则需要把它添加到属于它的当前（即第j个）灰度图像的斑点中，因为通过上面的距离比较可知，当前二值图像斑点属于当前灰度图像斑点
				if (!isNew)
				{
					//把当前二值图像斑点存入当前（即第j个）灰度图像斑点数组的最后位置
					centers[j].push_back(curCenters[i]);
					//得到构成该灰度图像斑点的所有二值图像斑点的数量
					size_t k = centers[j].size() - 1;
					//按照半径由小至大的顺序，把新得到的当前二值图像斑点放入当前灰度图像斑点数组的适当位置处，由于二值化阈值是按照从小到大的顺序设置，所以二值图像斑点本身就是按照面积的大小顺序被检测到的，因此此处的排序处理要相对简单一些
					while (k > 0 && centers[j][k].radius < centers[j][k - 1].radius)
					{
						centers[j][k] = centers[j][k - 1];
						k--;
					}
					centers[j][k] = curCenters[i];
					//由于当前二值图像斑点已经找到了属于它的灰度图像斑点，因此退出for循环，无需再遍历已有的灰度图像斑点
					break;
				}
			}
			if (isNew)    //当前二值图像斑点是新的斑点
			{
				//把当前斑点存入newCenters数组内
				newCenters.push_back(vector<Center>(1, curCenters[i]));
				//centers.push_back(vector<Center> (1, curCenters[i]));
			}
		}
		//把当前二值图像内的所有newCenters复制到centers内

		std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));

		//cout << " center_size:"<< centers.size() << endl;
#ifdef DEBUG_BLOB_DETECTOR
		//    imshow("binarized", keypointsImage );
		//waitKey();
#endif
	}    //所有二值图像斑点检测完毕
	//遍历所有灰度图像斑点，得到特征点信息
	cout << "Pro_Centers:" << centers.size() << endl;
	for (size_t i = 0; i < centers.size(); i++)
	{
		//如果属于当前灰度图像斑点的二值图像斑点的数量小于所设阈值，则该灰度图像斑点不是特征点
		if (centers[i].size() < params.minRepeatability)
			continue;
		Point2d sumPoint(0, 0);
		double normalizer = 0;
		//遍历属于当前灰度图像斑点的所有二值图像斑点
		for (size_t j = 0; j < centers[i].size(); j++)
		{
			sumPoint += centers[i][j].confidence * centers[i][j].location;    //公式2的分子
			normalizer += centers[i][j].confidence;    //公式2的分母
		}
		sumPoint *= (1. / normalizer);    //公式2，得到特征点的坐标位置
		//保存该特征点的坐标位置和尺寸大小
		KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius));
		keypoints.push_back(kpt);    //保存该特征点
	}


	//namedWindow("keypoints", CV_WINDOW_NORMAL);
	//Mat outImg;
	//if (outImg.channels() == 1)
	//	cvtColor(image, outImg, CV_GRAY2RGB);

	//for (size_t i = 0; i < keypoints.size(); i++)
	//{
	//	circle(outImg, keypoints[i].pt, keypoints[i].size, Scalar(255, 0, 255), -1);
	//}
	////drawKeypoints(image, keypoints, outImg);
	//imshow("keypoints", outImg);
	//waitKey();

#ifdef DEBUG_BLOB_DETECTOR
	namedWindow("keypoints", CV_WINDOW_NORMAL);
	Mat outImg = image.clone();
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		circle(outImg, keypoints[i].pt, keypoints[i].size, Scalar(255, 0, 255), -1);
	}
	//drawKeypoints(image, keypoints, outImg);
	imshow("keypoints", outImg);
	waitKey();
#endif
}

void circle_blob_test()
{
	ofstream fout1("./outputs/cam_corner.txt");
	ofstream fout2("./outputs/pro_corner.txt");

	ofstream fout("./outputs/cam_pro_result.txt");
	string file_out1 = "./outputs/cam_caliberation_result.txt";
	string file_out2 = "./outputs/pro_caliberation_result.txt";


	Size board_size = Size(dec_circle.board_col, dec_circle.board_row);    /* 标定板上每col、row的角点数 */

	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> cam_points_seq, cam_points_seq_1;
	vector<vector<Point2f>> pro_points_seq, pro_points_seq_1;

	int image_num;
	vector<Mat> white_edge_pic;
	cout << "Input the number of System Calibration：";
	cin >> image_num;
	cout << "Start Extracting Corners" << endl;
	char path[100] = "./calib_image/";
	char path1[100] = { 0 };
	char sub_name[20];
	bool not_flag[100] = { 0 };
	for (int i = 0; i < image_num; i++)
	{
		cout << "image_count = " << (i + 1) << endl;
		sprintf_s(sub_name, "%d_%d.bmp", (i + 1), cal_image_serial);
		strcpy(path1, path);
		strcat(path1, sub_name);

		Mat imageInput = cv::imread(path1, 0); //0: indicates read the gray image
		Mat imageOutput;
		GaussianBlur(imageInput, imageOutput, Size(5, 5), 3, 3);

		white_edge_pic.push_back(imageOutput);

		if (0 == cv::findCirclesGrid(255 - imageInput, board_size, image_points_buf, CALIB_CB_SYMMETRIC_GRID))
		{
			cout << "Image " << (i + 1) << " can not find chessboard corners!\n"; //找不到角点
			not_flag[i] = 1;
		}
		else
		{
			cam_points_seq.push_back(image_points_buf);
		}
	}

	int total = cam_points_seq.size();

	cout << "total the number of picture that corners can be detected = " << total << endl;
	int CornerNum = board_size.width * board_size.height;  //每张图片上总的角点数  

	int w_count = 0, r_count = 0;
	for (int i = 0; i < image_num; i++)
	{
		if (!not_flag[i])
		{
			vector<Point2f> image_points_buf1;  /* corners of projector */
			vector<Point2f> cam_points = cam_points_seq[w_count];
			image_points_buf1 = decode_corner_bilinear(cam_points, i + 1, CornerNum);
			pro_points_seq.push_back(image_points_buf1);
			w_count++;
		}
	}

	vector<Point2f> calib_points_2d(4);
	calib_points_2d[0] = Point2f(1.0 * dec_circle.d_x, 1.0 * dec_circle.d_y);
	calib_points_2d[1] = Point2f(dec_circle.board_col * dec_circle.d_x, 1.0 * dec_circle.d_y);
	calib_points_2d[2] = Point2f(dec_circle.board_col * dec_circle.d_x, dec_circle.board_row * dec_circle.d_y);
	calib_points_2d[3] = Point2f(1.0 * dec_circle.d_x, dec_circle.board_row * dec_circle.d_y);

	int factor = min(pro_width / 600, pro_height / (2 * 500));
	calib_points_2d[0] = Point2f(calib_points_2d[0].x * factor, calib_points_2d[0].y * factor);
	calib_points_2d[1] = Point2f(calib_points_2d[1].x * factor, calib_points_2d[1].y * factor);
	calib_points_2d[2] = Point2f(calib_points_2d[2].x * factor, calib_points_2d[2].y * factor);
	calib_points_2d[3] = Point2f(calib_points_2d[3].x * factor, calib_points_2d[3].y * factor);

	bool** D = new bool* [cam_height];
	Point2f** Pro_point = new Point2f * [cam_height];
	for (int i = 0; i < cam_height; i++)
	{
		D[i] = new bool[cam_width];
		Pro_point[i] = new Point2f[cam_width];
	}

	int count = 0;
	for (int i = 0; i < image_num; i++)
	{
		Mat input_image = white_edge_pic[i];

		if (!not_flag[i])
		{
			vector<Point2f> cam_points = cam_points_seq[count];
			vector<Point2f> pro_points = pro_points_seq[count];
			// compute for homography matrix H_pro
			vector<Point2f> tempImagePoint = pro_points;
			vector<Point2f> img_points_2d(4);
			img_points_2d[0] = tempImagePoint[0]; img_points_2d[0].y = img_points_2d[0].y / 2;
			img_points_2d[1] = tempImagePoint[10]; img_points_2d[1].y = img_points_2d[1].y / 2;
			img_points_2d[2] = tempImagePoint[98]; img_points_2d[2].y = img_points_2d[2].y / 2;
			img_points_2d[3] = tempImagePoint[88]; img_points_2d[3].y = img_points_2d[3].y / 2;
			Mat H_pro = findHomography(calib_points_2d, img_points_2d);

			decode_gp_double_calib(D, Pro_point, i + 1, path);

			//Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
			//detector->detect(255 - input_image, key_points1);
			vector<KeyPoint> key_points_cam, key_points_pro;

			SimpleBlob1 Sim_im;
			Sim_im.initialize();
			Sim_im.detectImpl(255 - input_image, key_points_cam, D);
			Sim_im.detectImpl_pro(255 - input_image, key_points_pro, Pro_point, D, H_pro);

			vector<Point2f> cam_circle_points, pro_circle_points;
			
			for (int k = 0; k < CornerNum; k++)
			{

				for (int j = 0; j < key_points_cam.size(); j++)
				{
					if (abs(key_points_cam[j].pt.x - cam_points[k].x) < 5 && abs(key_points_cam[j].pt.y - cam_points[k].y) < 5)
					{
						cam_circle_points.push_back(key_points_cam[j].pt);
					}
				}

				for (int j = 0; j < key_points_pro.size(); j++)
				{
					if (abs(key_points_pro[j].pt.x - pro_points[k].x) < 5 && abs(key_points_pro[j].pt.y * 2 - pro_points[k].y) < 5)
					{
						Point2f p_temp = key_points_pro[j].pt;
						p_temp.y = p_temp.y * 2;
						pro_circle_points.push_back(p_temp);
						break;
					}
				}
			}

			cam_points_seq_1.push_back(cam_circle_points);
			pro_points_seq_1.push_back(pro_circle_points);


		}

		count = count + 1;
	}


	{
		//drawKeypoints(imageInput, key_points_cam, output_img, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		//namedWindow("SimpleBlobDetector", WINDOW_NORMAL);
		//imshow("SimpleBlobDetector", output_img);
		//waitKey(0);
		// cv::destroyAllWindows();

	}



	Mat cameraMatrix1 = Mat(3, 3, CV_32FC1, Scalar::all(0));
	Mat distCoeffs1 = Mat(1, 5, CV_32FC1, Scalar::all(0));

	Mat cameraMatrix2 = Mat(3, 3, CV_32FC1, Scalar::all(0));
	Mat distCoeffs2 = Mat(1, 5, CV_32FC1, Scalar::all(0));

	Size cam_size = Size(cam_width, cam_height);
	Size pro_size = Size(pro_width, 2 * pro_height);

	/*棋盘三维信息*/
	Size square_size = Size(dec_circle.d_y, dec_circle.d_x);  /* 实际测量得到的标定板上每个棋盘格的大小 */
	vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
	int i, j, t;
	for (t = 0; t < total; t++)
	{
		vector<Point3f> tempPointSet;
		for (j = 0; j < board_size.height; j++)
		{
			for (i = 0; i < board_size.width; i++)
			{
				Point3f realPoint;
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}


	cout << "*********** Start Camera Calibration and Projector Calibration ***********" << endl;

	vector<Mat> tvecsMat1, tvecsMat2;  /* 每幅图像的旋转向量 */
	vector<Mat> rvecsMat1, rvecsMat2; /* 每幅图像的平移向量 */

	cv::calibrateCamera(object_points, cam_points_seq, cam_size, cameraMatrix1, distCoeffs1, rvecsMat1, tvecsMat1, CALIB_FIX_K3);
	cv::calibrateCamera(object_points, pro_points_seq_1, pro_size, cameraMatrix2, distCoeffs2, rvecsMat2, tvecsMat2, CALIB_FIX_K3);

	//Linear Model of Projector
	//cv::calibrateCamera(object_points, pro_points_seq, pro_size, cameraMatrix2, distCoeffs2 , rvecsMat2, tvecsMat2, 0, TermCriteria(
	//	TermCriteria::COUNT + TermCriteria::EPS, 0, DBL_EPSILON));
	// distCoeffs2 = Mat(1, 5, CV_32FC1, Scalar::all(0));

	compute_save_result(object_points, cam_points_seq, cameraMatrix1, distCoeffs1, rvecsMat1, tvecsMat1, file_out1, total, board_size, 0);
	compute_save_result(object_points, pro_points_seq_1, cameraMatrix2, distCoeffs2, rvecsMat2, tvecsMat2, file_out2, total, board_size, 1);

	Mat rvec, tvec, E, F;
	double rms = stereoCalibrate(object_points, cam_points_seq, pro_points_seq_1, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cam_size, rvec, tvec, E, F);

	// save calibration results to parameter.xml
	FileStorage fs("./outputs/parameter.xml", FileStorage::WRITE);

	if (fs.isOpened())
	{
		fs << "cameraMatrix1" << cameraMatrix1 << "distCoeffs1" << distCoeffs1;
		fs << "cameraMatrix2" << cameraMatrix2 << "distCoeffs2" << distCoeffs2;
		fs << "R" << rvec << "T" << tvec;
	}
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix1 << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs1 << endl << endl << endl;
	fout << "投影仪内参数矩阵：" << endl;
	fout << cameraMatrix2 << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs2 << endl << endl << endl;
	fout << "相机-投影仪相对位置关系（相机世界坐标系）" << endl;
	fout << rvec << endl;
	fout << tvec << endl;

	fout.close();
	fout1.close();
	fout2.close();
	cout << "Calbration Done!Press Any Key to exit! " << endl;
	cin.ignore();
	cin.ignore();

}