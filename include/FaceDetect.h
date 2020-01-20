#ifndef FACE_DETECT_H
#define FACE_DETECT_H
#include <iostream>
#include "mtcnn.h"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
struct FaceData
{
	int num;
	float *location;
	float *keypoint;
};
void FaceDate_free(FaceData *data);
void FaceDetect(Mat frame, mtcnn *detector, struct FaceData *FaceInfo);

#endif
