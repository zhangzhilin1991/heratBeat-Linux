#include"FaceDetect.h"

void FaceDate_free(FaceData *data) {
	free(data->keypoint);
	free(data->location);
	data->num = 0;
}

void FaceDetect(Mat frame, mtcnn *detector,struct FaceData *FaceInfo) {
	
	//check the size of init
	if (frame.rows != detector->init_row || frame.cols != detector->init_col) {
		cout << "Please check the size of input image" << endl;
		return;
	}
	detector->findFace(frame);
	FaceInfo->num = detector->facenum;
	FaceInfo->location = (float *)malloc(FaceInfo->num * 4 * sizeof(float));
	FaceInfo->keypoint = (float *)malloc(FaceInfo->num * 10 * sizeof(float));
	float *p_local = FaceInfo->location;
	float *p_keyp = FaceInfo->keypoint;
	for (vector<struct Bbox>::iterator it = detector->thirdBbox_.begin(); it != detector->thirdBbox_.end(); it++) {
		if ((*it).exist) {
			*p_local++ = (*it).y1;
			*p_local++ = (*it).x1;
			*p_local++ = (*it).y2;
			*p_local++ = (*it).x2;
			memcpy(p_keyp, it->ppoint, 10*sizeof(mydataFmt));
			p_keyp += 10;
			//rectangle(frame, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0, 0, 255), 2, 8, 0);
			//for(int num=0;num<5;num++)circle(image,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
		}
	}
	detector->thirdBbox_.clear();
}
