//#pragma once
#ifndef JS_UTILS
#define JS_UTILS
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

#include "RPPG.hpp"
#include "NumCpp.hpp"

using namespace cv;

#ifndef M_PI
#define M_PI 3.1415926
#endif

int get_len(nc::NdArray<double> input);

double get_hamming(double n, double M);

nc::NdArray<double> hamming(double M);

vector<double> get_angle(vector<complex<double>> raw);

vector<double> get_abs(vector<complex<double>> raw);

void delete_rows(Mat &src);

#endif