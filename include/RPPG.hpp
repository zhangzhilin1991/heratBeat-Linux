//
//  RPPG.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 7/07/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef RPPG_hpp
#define RPPG_hpp

#include <fstream>
#include <string>
#include <deque>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

#include "mtcnn.h"
#include "NumCpp.hpp"
#include "FaceDetect.h"

#include <stdio.h>


using namespace cv;
using namespace dnn;
using namespace std;

enum rPPGAlgorithm { g, pca, xminay };
enum faceDetAlgorithm { haar, deep, mtcnn_deep };
static double updating_mask[4] = {0.1, 0.2, 0.4, 0.5};

class RPPG {

public:

    // Constructor
    RPPG() {;}

    // Load Settings
    bool load(const rPPGAlgorithm rPPGAlg, const faceDetAlgorithm faceDetAlg,
              const int width, const int height, const double timeBase, const int downsample,
              const double samplingFrequency, const double rescanFrequency,
              const int minSignalSize, const int maxSignalSize,
              const string &logPath, const string &haarPath,
              const string &dnnProtoPath, const string &dnnModelPath,
              const bool log, const bool gui, int64_t buffer_size);
	
	bool constantLoad();

    void processFrame(Mat &frameRGB, Mat &frameGray, int time);
	void processFrame(Mat &frameRGB, Mat &frameGray, int time, mtcnn &mtcnn_find);
	void countFrame();

    void exit();
	mtcnn face_detector = mtcnn(480, 640);
    typedef vector<Point2f> Contour2f;
    typedef nc::NdArray<double> Ndarrayd;

private:

	
    void detectFace(Mat &frameRGB, Mat &frameGray);
	void detectFace(Mat &frameRGB, Mat &frameGray, mtcnn &mtcnn_find);
    void setNearestBox(vector<Rect> boxes);
    void detectCorners(Mat &frameGray);
    void trackFace(Mat &frameGray);
    void updateMask(Mat &frameGray);
    void updateROI();
    void extractSignal_g();
    void extractSignal_pca();
    void extractSignal_xminay();
    void estimateHeartrate(Mat &frameRGB, Mat1b &mask, Mat1d &output_s, Mat1d &output_bpms, double *output_meanBpms);
    void draw(Mat &frameRGB);
    void invalidateFace();
    void log();

	//void js(Mat &frameRGB, Rect &roi);

    // The algorithm
    rPPGAlgorithm rPPGAlg;

    // The classifier
    faceDetAlgorithm faceDetAlg;
    CascadeClassifier haarClassifier;
    Net dnnClassifier;

    // Settings
    Size minFaceSize;
    int maxSignalSize;
    int minSignalSize;
    double rescanFrequency;
    double samplingFrequency;
    double timeBase;
    bool logMode;
    bool guiMode;

    // State variables
    int64_t time;
    double fps;
    int high;
    int64_t lastSamplingTime;
    int64_t lastScanTime;
    int low;
    int64_t now;
    bool faceValid;
    bool rescanFlag;

    // Tracking
    Mat lastFrameGray;
    Contour2f corners;

    // Mask
    Rect box;
    Mat1b mask;

    Rect roi;

    // Raw signal
    Mat1d s;
    Mat1d t;
    Mat1b re;

    // 心跳检测
	// 缓存大小

	int64_t buffer_size;
	int64_t output_dim;
	Ndarrayd fft;
	Ndarrayd freqs;
	double bpm;
	
    // Estimation
    Mat1d s_f;
    Mat1d bpms;
    Mat1d powerSpectrum;

	Mat1d bpms_forehead;
	Mat1d bpms_Rface;
	Mat1d bpms_Lface;


	double meanBpm;
	//临时储存

    double minBpm;
    double maxBpm;

    // Logfiles
	int framecount;
    ofstream logfile;
    ofstream logfileDetailed;
    string logfilepath;

	// keyPoints
	vector<Point> landmarks = {};
	Point last_face;
	bool new_face;

};

#endif /* RPPG_hpp */