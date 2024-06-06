// stdafx.h : 标准系统包含文件的包含文件，

#if defined(_WIN32) || defined(_WIN64)

#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <tchar.h>
#include <windows.h>

#elif defined(MACOSX)

#else

#include <stdio.h>

#endif

#pragma once  //stdafx只编译一次
#pragma warning(disable:4996)

// TODO：define global variable
#define PI 3.1415926

// TODO: define wether projecting white and black patterns
#define Project_BW 0

#define Project_CROSS 1

#include "math.h"
#include "string.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp" 

//#include "pcl/visualization/pcl_visualizer.h"

#include <iostream>  
#include <fstream>
#include <iomanip>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace cv;
using namespace std;

//Point2d(x,y) == y represents row, x represents col
//Size(width, height)

struct decode_pattern {
	int phase_num;
	int cycle;
	double threshold;
	double phase_offset;
	int board_row;
	int board_col;
	double d_y;
	double d_x;
	int grey_thresh;	//used to find decodable region D_grey using grey-coded patterns
	int phase_thresh;	//used to find decodable region D_phase using phase-shifted patterns
	int search_width;	//the width of searching window which is used to correct the unwrapping phase errors
};

const int cam_width = 1280, cam_height = 1024;
const int pro_width = 912, pro_height = 1140;


const decode_pattern dec_circle = { 6, 20, 0.5, 3.14, 9, 11, 50, 50, 0 , 0, 0};// Circle ChessBoard



//the serial number of projecting white light
const int cal_image_serial = ceil(log2(pro_width / dec_circle.cycle)) + ceil(log2(pro_height / dec_circle.cycle)) + 2; // the serial number of image which is used to extract corners
const int single_image_serial = ceil(log2(pro_width / dec_config.cycle)) + 2;
const int double_image_serial = ceil(log2(pro_width / dec_config.cycle)) + ceil(log2(pro_height / dec_config.cycle)) + 2;

const char path_calib_image[100] = "./calib_image/";
const char path_decode_image[100] = "./decode_image/";
const char path_decode_image_cross[100] = "./decode_image_cross/";
const char path_outputs[100] = "./outputs/";

const int grey_col = ceil(log2(double(pro_width) / double(dec_config.cycle)));
const int grey_row = ceil(log2(double(pro_height) / double(dec_config.cycle)));

const int w_i = ceil(double(pro_width) / double(dec_config.cycle));
const int h_i = ceil(double(pro_height) / double(dec_config.cycle));
const int N_w = ceil(log2(w_i));
const int N_h = ceil(log2(h_i));
const int offset_w = floor((pow(2, N_w) - w_i) / 2);
const int offset_h = floor((pow(2, N_h) - h_i) / 2);



