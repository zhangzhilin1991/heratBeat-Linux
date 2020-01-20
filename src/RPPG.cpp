//
//  RPPG.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 7/07/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "RPPG.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

#include "opencv.hpp"
#include <cmath>
#include "js_utils.h"

using namespace cv;
using namespace dnn;
using namespace std;

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25

bool RPPG::load(const rPPGAlgorithm rPPGAlg, const faceDetAlgorithm faceDetAlg,
                const int width, const int height, const double timeBase, const int downsample,
                const double samplingFrequency, const double rescanFrequency,
                const int minSignalSize, const int maxSignalSize,
                const string &logPath, const string &haarPath,
                const string &dnnProtoPath, const string &dnnModelPath,
                const bool log, const bool gui, int64_t buffer_size) {

    this->rPPGAlg = rPPGAlg;
    this->faceDetAlg = faceDetAlg;
    this->guiMode = gui;
    this->lastSamplingTime = 0;
    this->logMode = log;
    this->minFaceSize = Size(min(width, height) * REL_MIN_FACE_SIZE, min(width, height) * REL_MIN_FACE_SIZE);
    this->maxSignalSize = maxSignalSize;
    this->minSignalSize = minSignalSize;
    this->rescanFlag = false;
    this->rescanFrequency = rescanFrequency;
    this->samplingFrequency = samplingFrequency;
    this->timeBase = timeBase;
	// 新增加的初始化
    this->buffer_size = buffer_size;
	this->framecount = 0;
	this->new_face = true;

    // Load classifier 不采用下述两种检测器，故注释
    switch (faceDetAlg) {
      case haar:
        //haarClassifier.load(haarPath);
        break;
      case deep:
        //dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);
        break;
    }

    // Setting up logfilepath
    ostringstream path_1;
    path_1 << logPath << "_rppg=" << rPPGAlg << "_facedet=" << faceDetAlg << "_min=" << minSignalSize << "_max=" << maxSignalSize << "_ds=" << downsample;
    this->logfilepath = path_1.str();

    // Logging bpm according to sampling frequency
    std::ostringstream path_2;
    path_2 << logfilepath << "_meanbpm.csv";
    logfile.open(path_2.str());
	logfile << "frame,time,mean,\n";
    logfile.flush();

    // Logging bpm detailed
    std::ostringstream path_3;
    path_3 << logfilepath << "_bpmAll.csv";
    logfileDetailed.open(path_3.str());
	logfileDetailed << "frame,time,bpm,fps,\n";
    logfileDetailed.flush();

    return true;
}

bool RPPG::constantLoad()
{
	// this->rPPGAlg = rPPGAlg;
	this->faceDetAlg = mtcnn_deep;
	this->guiMode = true;
	this->lastSamplingTime = 0;
	this->logMode = false;
	this->minFaceSize = Size(min(640, 480) * REL_MIN_FACE_SIZE, min(640, 480) * REL_MIN_FACE_SIZE);
	this->maxSignalSize = 5;
	this->minSignalSize = 5;
	this->rescanFlag = false;
	this->rescanFrequency = 1;
	this->samplingFrequency = 1;
	this->timeBase = 0.001;
	this->buffer_size = 250;

	// Load classifier
	switch (faceDetAlg) {
	case haar:
		//haarClassifier.load(haarPath);
		break;
	case deep:
		//dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);
		break;
	}

	return true;
}

void RPPG::exit() {
    logfile.close();
    logfileDetailed.close();
}

// 该方法没有被调用，可忽略
void RPPG::processFrame(Mat &frameRGB, Mat &frameGray, int time) {

    // Set time
    this->time = time;

    if (!faceValid) {

        cout << "Not valid, finding a new face" << endl;

        lastScanTime = time;
        detectFace(frameRGB, frameGray);

    } else if ((time - lastScanTime) * timeBase >= 1/rescanFrequency) {

        cout << "Valid, but rescanning face" << endl;

        lastScanTime = time;
        detectFace(frameRGB, frameGray);
        rescanFlag = true;

    } else {

        cout << "Tracking face" << endl;

        trackFace(frameGray);
    }

    if (faceValid) {

        // Update fps
        fps = getFps(t, timeBase);

        // Remove old values from raw signal buffer
        while (s.rows > fps * maxSignalSize) {
            push(s);
            push(t);
            push(re);
        }

        assert(s.rows == t.rows && s.rows == re.rows);

        // New values
        Scalar means = mean(frameRGB, mask);
        // Add new values to raw signal buffer
        double values[] = {means(0), means(1), means(2)};
        s.push_back(Mat(1, 3, CV_64F, values));
        t.push_back(time);

        // Save rescan flag
        re.push_back(rescanFlag);

        // Update fps
        fps = getFps(t, timeBase);

        // Update band spectrum limits
        low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
        high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1;

        // If valid signal is large enough: estimate
        if (s.rows >= fps * minSignalSize) {

            // Filtering
            switch (rPPGAlg) {
                case g:
                    extractSignal_g();
                    break;
                case pca:
                    extractSignal_pca();
                    break;
                case xminay:
                    extractSignal_xminay();
                    break;
            }

            // HR estimation
            // estimateHeartrate();
        }

        if (guiMode) {
            draw(frameRGB);
        }
    }

    rescanFlag = false;

    frameGray.copyTo(lastFrameGray);
}

// 检测心率的主要流程如下
void RPPG::processFrame(Mat &frameRGB, Mat &frameGray, int time, mtcnn &mtcnn_find) {

	// Set time
	this->time = time;

	// 从输入图像中检测人脸
	if (!faceValid) {

		cout << "Not valid, finding a new face" << endl;

		lastScanTime = time;
		//detectFace(frameRGB, frameGray);
		detectFace(frameRGB, frameGray, mtcnn_find);

	}
	// 有人脸，且当前时间距离上次采样的时间差超过1s，则重新采样(现在改为0.5s)
	else if ((time - lastScanTime) * timeBase >= 0.5 / rescanFrequency) {

		cout << "Valid, but rescanning face" << endl;

		lastScanTime = time;

		detectFace(frameRGB, frameGray, mtcnn_find);
		rescanFlag = true;

	}
	else {
		// 有人脸，当前时间与上次采样时间小于0.5s，则采用追踪
		cout << "Tracking face" << endl;

		trackFace(frameGray);
	}

	// 检测心率
	if (faceValid) {

		assert(s.rows == t.rows && s.rows == re.rows);
		
		t.push_back(time);
		re.push_back(rescanFlag);

		// 检测心率入口
		estimateHeartrate(frameRGB, mask, s, bpms, &meanBpm);
		
		if ((time - lastSamplingTime) * timeBase >= 0.5 / samplingFrequency) {
			lastSamplingTime = time;

			meanBpm = mean(bpms)(0);
		}
        if (guiMode) {
            draw(frameRGB);
        }
		if (logMode)
		{
			log();
		}
	}

	rescanFlag = false;

	frameGray.copyTo(lastFrameGray);
}

// 计算当前帧数
void RPPG::countFrame()
{
	this->framecount = framecount + 1;
}


// 该方法没有调用，可忽视
void RPPG::detectFace(Mat &frameRGB, Mat &frameGray) {

    cout << "Scanning for faces..." << endl;
    vector<Rect> boxes = {};

    switch (faceDetAlg) {
      case haar:
        // Detect faces with Haar classifier
        haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CASCADE_SCALE_IMAGE, minFaceSize);
        break;
      case deep:
        // Detect faces with DNN
        Mat resize300;
        cv::resize(frameRGB, resize300, Size(300, 300));
        Mat blob = blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
        dnnClassifier.setInput(blob);
        Mat detection = dnnClassifier.forward();
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        float confidenceThreshold = 0.5;

        for (int i = 0; i < detectionMat.rows; i++) {
          float confidence = detectionMat.at<float>(i, 2);
          if (confidence > confidenceThreshold) {
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frameRGB.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frameRGB.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frameRGB.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frameRGB.rows);
            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
            boxes.push_back(object);
          }
        }
        break;
    }

    if (boxes.size() > 0) {

        cout << "Found a face" << endl;

        setNearestBox(boxes);
        detectCorners(frameGray);
        updateROI();
        updateMask(frameGray);
        faceValid = true;

    } else {

        cout << "Found no face" << endl;
        invalidateFace();
    }
}

// 检测人脸的主要代码， 运行case mtcnn_deep
void RPPG::detectFace(Mat &frameRGB, Mat &frameGray, mtcnn &mtcnn_find) {

	cout << "Scanning for faces..." << endl;
	
	vector<Rect> boxes = {};

	switch (faceDetAlg) {
	case haar:
	{
		// Detect faces with Haar classifier
		haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CASCADE_SCALE_IMAGE, minFaceSize);
		break;
	}
	case deep:
	{
		// Detect faces with DNN
		Mat resize300;
		cv::resize(frameRGB, resize300, Size(300, 300));
		Mat blob = blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
		dnnClassifier.setInput(blob);
		Mat detection = dnnClassifier.forward();
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		float confidenceThreshold = 0.5;

		for (int i = 0; i < detectionMat.rows; i++) {
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold) {
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frameRGB.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frameRGB.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frameRGB.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frameRGB.rows);
				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));
				boxes.push_back(object);
			}
		}
		break;
	}
	case mtcnn_deep:
	{
		FaceData FaceInfo;

		FaceDetect(frameRGB, &mtcnn_find, &FaceInfo);
		float* p_local = FaceInfo.location; 
		int maxArg = 0;
		int maxNum = 0;
		vector<Rect> tmp_boxes;
		vector<Point> tmp_landmarks;
		if (new_face)
		{
			for (int i = 0; i < FaceInfo.num; i++) {
				int x1 = *p_local++;
				int y1 = *p_local++;
				int x2 = *p_local++;
				int y2 = *p_local++;
				float* ppoint = FaceInfo.keypoint + i * 10;
				double fos_ratio = (x2 - x1) * (y2 - y1);
				double ftc_distance = abs((x2 + x1) / 2.0 - 320) + abs((y2 + y1) / 2.0 - 240);
				double face_weight = 0.8 * fos_ratio + 0.2 * ftc_distance;
				if (face_weight > maxNum)
				{
					maxNum = face_weight;
					maxArg = i;
				}

				Rect object(x1, y1, (x2 - x1), (y2 - y1));
				tmp_boxes.push_back(object);

				//画关键点
				for (int num = 0; num < 5; num++) {
					Point landmark = Point((int) * (ppoint + num), (int) * (ppoint + num + 5));
					tmp_landmarks.push_back(landmark);
				}
			}
		}
		else
		{
			maxNum = 9999;
			for (int i = 0; i < FaceInfo.num; i++) {
				int x1 = *p_local++;
				int y1 = *p_local++;
				int x2 = *p_local++;
				int y2 = *p_local++;
				float* ppoint = FaceInfo.keypoint + i * 10;
				double point_distance = abs(*(ppoint + 2) - last_face.x) + abs(*(ppoint + 7) - last_face.y);
				double face_weight = point_distance;
				if (face_weight < maxNum)
				{
					maxNum = face_weight;
					maxArg = i;
				}

				Rect object(x1, y1, (x2 - x1), (y2 - y1));
				tmp_boxes.push_back(object);

				//画关键点
				for (int num = 0; num < 5; num++) {
					Point landmark = Point((int) * (ppoint + num), (int) * (ppoint + num + 5));
					tmp_landmarks.push_back(landmark);
				}
			}
		}
		
		if (tmp_boxes.size() > 0)
		{
			boxes.push_back(tmp_boxes.at(maxArg));
			for (int tmp = 0; tmp < 5; tmp++)
			{
				landmarks.push_back(tmp_landmarks.at(maxArg * 5 + tmp));
			}
			if (new_face)
			{
				new_face = !new_face;
			}
			last_face = landmarks.at(2);
		}
		else
		{
			new_face = true;
		}
		break;
	}
	}

	if (boxes.size() > 0) {

		cout << "Found a face" << endl;

		setNearestBox(boxes);
		detectCorners(frameGray);
		updateROI();
		updateMask(frameGray);
		faceValid = true;

	}
	else {

		cout << "Found no face" << endl;
		invalidateFace();
	}
}

void RPPG::setNearestBox(vector<Rect> boxes) {
    int index = 0;
    Point p = box.tl() - boxes.at(0).tl();
    int min = p.x * p.x + p.y * p.y;
    for (int i = 1; i < boxes.size(); i++) {
        p = box.tl() - boxes.at(i).tl();
        int d = p.x * p.x + p.y * p.y;
        if (d < min) {
            min = d;
            index = i;
        }
    }
    box = boxes.at(index);
}

void RPPG::detectCorners(Mat &frameGray) {

    // Define tracking region
    Mat trackingRegion = Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
    Point points[1][4];
    points[0][0] = Point(box.tl().x + 0.22 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][1] = Point(box.tl().x + 0.78 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][2] = Point(box.tl().x + 0.70 * box.width,
                         box.tl().y + 0.65 * box.height);
    points[0][3] = Point(box.tl().x + 0.30 * box.width,
                         box.tl().y + 0.65 * box.height);
    const Point *pts[1] = {points[0]};
    int npts[] = {4};
    fillPoly(trackingRegion, pts, npts, 1, WHITE);

    // Apply corner detection
    goodFeaturesToTrack(frameGray,
                        corners,
                        MAX_CORNERS,
                        QUALITY_LEVEL,
                        MIN_DISTANCE,
                        trackingRegion,
                        3,
                        false,
                        0.04);
}

void RPPG::trackFace(Mat &frameGray) {

    // Make sure enough corners are available
    if (corners.size() < MIN_CORNERS) {
        detectCorners(frameGray);
    }

    Contour2f corners_1;
    Contour2f corners_0;
    vector<uchar> cornersFound_1;
    vector<uchar> cornersFound_0;
    Mat err;

	// 人脸旋转超过幅度，角点为空，光流计算将会报错
	if (corners.empty()) {
		cout << "ERROR====" << endl;
		return;
	}

    // Track face features with Kanade-Lucas-Tomasi (KLT) algorithm
    calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);

    // Backtrack once to make it more robust
    calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);

    // Exclude no-good corners
    Contour2f corners_1v;
    Contour2f corners_0v;
    for (size_t j = 0; j < corners.size(); j++) {
        if (cornersFound_1[j] && cornersFound_0[j]
            && norm(corners[j]-corners_0[j]) < 2) {
            corners_0v.push_back(corners_0[j]);
            corners_1v.push_back(corners_1[j]);
        } else {
            cout << "Mis!" << std::endl;
        }
    }

    if (corners_1v.size() >= MIN_CORNERS) {

        // Save updated features
        corners = corners_1v;

        // Estimate affine transform
        Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);

        if (transform.total() > 0) {
            // Update box
            Contour2f boxCoords;
            boxCoords.push_back(box.tl());
            boxCoords.push_back(box.br());
            Contour2f transformedBoxCoords;

            cv::transform(boxCoords, transformedBoxCoords, transform);
            box = Rect(transformedBoxCoords[0], transformedBoxCoords[1]);

            // Update roi
            Contour2f roiCoords;
            roiCoords.push_back(roi.tl());
            roiCoords.push_back(roi.br());
            Contour2f transformedRoiCoords;
            cv::transform(roiCoords, transformedRoiCoords, transform);
            roi = Rect(transformedRoiCoords[0], transformedRoiCoords[1]);

			updateMask(frameGray);
		}

	}
 else {
	 cout << "Tracking failed! Not enough corners left." << endl;

	 //
	 /*invalidateFace();*/
	}
}

void RPPG::updateROI() {

	// 根据关键点判断脸的朝向
	if (landmarks.size() < 5) {
		this->roi = Rect(Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
			Point(box.tl().x + 0.6 * box.width, box.tl().y + 0.25 * box.height));
	}
	else
	{
		// 生成候选框
		// 人眼中心到人脸上边框的距离
		int top_height = (landmarks[0].y + landmarks[1].y) / 2 - box.tl().y;

		// 两眼之间的距离
		int eye_width = landmarks[1].x - landmarks[0].x;

		// 额头的框
		roi = Rect(landmarks[0].x + eye_width / 5, landmarks[0].y - 0.8 * top_height, 0.6 * eye_width, 0.3 * top_height);
	}

	this->landmarks.clear();
}

void RPPG::updateMask(Mat &frameGray) {

    //cout << "Update mask" << endl;

    mask = Mat::zeros(frameGray.size(), frameGray.type());
    rectangle(mask, this->roi, WHITE, FILLED);
}

void RPPG::invalidateFace() {

	cout << "Invalidation !" << endl;
    s = Mat1d();

    t = Mat1d();
    re = Mat1b();
    faceValid = false;
}

void RPPG::extractSignal_g() {

	// Denoise
	Mat s_den = Mat(s.rows, 1, CV_64F);
	denoise(s.col(1), re, s_den);

	// Normalise
	normalization(s_den, s_den);

	// Detrend
	Mat s_det = Mat(s_den.rows, s_den.cols, CV_64F);
	detrend(s_den, s_det, fps);

	// Moving average
	Mat s_mav = Mat(s_det.rows, s_det.cols, CV_64F);
	movingAverage(s_det, s_mav, 3, fmax(floor(fps / 6), 2));

	s_mav.copyTo(s_f);

	// Logging
	if (logMode) {
		std::ofstream log;
		std::ostringstream filepath;
		filepath << logfilepath << "_signal_" << time << ".csv";
		log.open(filepath.str());
		log << "re;g;g_den;g_det;g_mav\n";
		for (int i = 0; i < s.rows; i++) {
			log << re.at<bool>(i, 0) << ";";
			log << s.at<double>(i, 1) << ";";
			log << s_den.at<double>(i, 0) << ";";
			log << s_det.at<double>(i, 0) << ";";
			log << s_mav.at<double>(i, 0) << "\n";
		}
		log.close();
	}
}

// 该方法不调用 可删去
void RPPG::extractSignal_pca() {

	// Denoise signals
	Mat s_den = Mat(s.rows, s.cols, CV_64F);
	denoise(s, re, s_den);

	// Normalize signals
	normalization(s_den, s_den);

	// Detrend
	Mat s_det = Mat(s.rows, s.cols, CV_64F);
	detrend(s_den, s_det, fps);

	// PCA to reduce dimensionality
	Mat s_pca = Mat(s.rows, 1, CV_32F);
	Mat pc = Mat(s.rows, s.cols, CV_32F);
	pcaComponent(s_det, s_pca, pc, low, high);

	// Moving average
	Mat s_mav = Mat(s.rows, 1, CV_32F);
	movingAverage(s_pca, s_mav, 3, fmax(floor(fps / 6), 2));

	s_mav.copyTo(s_f);

	// Logging
	if (logMode) {
		std::ofstream log;
		std::ostringstream filepath;
		filepath << logfilepath << "_signal_" << time << ".csv";
		log.open(filepath.str());
		log << "re;r;g;b;r_den;g_den;b_den;r_det;g_det;b_det;pc1;pc2;pc3;s_pca;s_mav\n";
		for (int i = 0; i < s.rows; i++) {
			log << re.at<bool>(i, 0) << ";";
			log << s.at<double>(i, 0) << ";";
			log << s.at<double>(i, 1) << ";";
			log << s.at<double>(i, 2) << ";";
			log << s_den.at<double>(i, 0) << ";";
			log << s_den.at<double>(i, 1) << ";";
			log << s_den.at<double>(i, 2) << ";";
			log << s_det.at<double>(i, 0) << ";";
			log << s_det.at<double>(i, 1) << ";";
			log << s_det.at<double>(i, 2) << ";";
			log << pc.at<double>(i, 0) << ";";
			log << pc.at<double>(i, 1) << ";";
			log << pc.at<double>(i, 2) << ";";
			log << s_pca.at<double>(i, 0) << ";";
			log << s_mav.at<double>(i, 0) << "\n";
		}
		log.close();
	}
}

// 该方法不调用 可删去
void RPPG::extractSignal_xminay() {

	// Denoise signals
	Mat s_den = Mat(s.rows, s.cols, CV_64F);
	denoise(s, re, s_den);

	// Normalize raw signals
	Mat s_n = Mat(s_den.rows, s_den.cols, CV_64F);
	normalization(s_den, s_n);

	// Calculate X_s signal
	Mat x_s = Mat(s.rows, s.cols, CV_64F);
	addWeighted(s_n.col(0), 3, s_n.col(1), -2, 0, x_s);

	// Calculate Y_s signal
	Mat y_s = Mat(s.rows, s.cols, CV_64F);
	addWeighted(s_n.col(0), 1.5, s_n.col(1), 1, 0, y_s);
	addWeighted(y_s, 1, s_n.col(2), -1.5, 0, y_s);

	// Bandpass
	Mat x_f = Mat(s.rows, s.cols, CV_32F);
	bandpass(x_s, x_f, low, high);
	x_f.convertTo(x_f, CV_64F);
	Mat y_f = Mat(s.rows, s.cols, CV_32F);
	bandpass(y_s, y_f, low, high);
	y_f.convertTo(y_f, CV_64F);

	// Calculate alpha
	Scalar mean_x_f;
	Scalar stddev_x_f;
	meanStdDev(x_f, mean_x_f, stddev_x_f);
	Scalar mean_y_f;
	Scalar stddev_y_f;
	meanStdDev(y_f, mean_y_f, stddev_y_f);
	double alpha = stddev_x_f.val[0] / stddev_y_f.val[0];

	// Calculate signal
	Mat xminay = Mat(s.rows, 1, CV_64F);
	addWeighted(x_f, 1, y_f, -alpha, 0, xminay);

	// Moving average
	movingAverage(xminay, s_f, 3, fmax(floor(fps / 6), 2));

	// Logging
	if (logMode) {
		std::ofstream log;
		std::ostringstream filepath;
		filepath << logfilepath << "_signal_" << time << ".csv";
		log.open(filepath.str());
		log << "r;g;b;r_den;g_den;b_den;x_s;y_s;x_f;y_f;s;s_f\n";
		for (int i = 0; i < s.rows; i++) {
			log << s.at<double>(i, 0) << ";";
			log << s.at<double>(i, 1) << ";";
			log << s.at<double>(i, 2) << ";";
			log << s_den.at<double>(i, 0) << ";";
			log << s_den.at<double>(i, 1) << ";";
			log << s_den.at<double>(i, 2) << ";";
			log << x_s.at<double>(i, 0) << ";";
			log << y_s.at<double>(i, 0) << ";";
			log << x_f.at<double>(i, 0) << ";";
			log << y_f.at<double>(i, 0) << ";";
			log << xminay.at<double>(i, 0) << ";";
			log << s_f.at<double>(i, 0) << "\n";
		}
		log.close();
	}
}

void RPPG::estimateHeartrate(Mat &frameRGB, Mat1b &input_mask, Mat1d &output_s, Mat1d &output_bpms, double *output_meanBpms) {

	logfileDetailed << framecount << ",";
	logfileDetailed << time << ",";

	logfile << framecount << ",";
	logfile << time << ",";
	logfile << meanBpm << ",";
	logfile << ",";


	Scalar means = mean(frameRGB, input_mask);
	double values = (means(0) + means(1) + means(2)) / 3.0;

	output_s.push_back(Mat(1, 1, CV_64F, &values));
	
	int L = output_s.rows;
	if (L > 250)
	{
		delete_rows(output_s);
	}
	if (t.rows > 250)
	{
		delete_rows(t);
	}
	if (re.rows > 250)
	{
		delete_rows(re);
	}

	if (L > 10) {
		nc::NdArray<double> s_array = nc::NdArray<double>(output_s.reshape(0, 1));
		nc::NdArray<double> t_array = nc::NdArray<double>(t.reshape(0, 1));
		nc::NdArray<double> interpolated;

		output_dim = output_s.rows;
		fps = getFps(t, timeBase);
		//cout << "size of t_array" << t_array.size() << endl;
		nc::NdArray<double> even_times = nc::linspace<double>(t_array.front(), t_array.back(), L);

		if (s.rows == s_array.numCols()) {
			interpolated = nc::interp(even_times, t_array, s_array);
		}
		else
		{
			return;
		}
		interpolated = hamming(L) * interpolated;
		interpolated = interpolated - *interpolated.mean().data();

		vector<complex<double>> raw;
		cv::dft(interpolated.toStlVector(), raw, DFT_COMPLEX_OUTPUT);
		raw.resize(ceil(raw.size() / 2.0));
		// 有待测试
		// raw.resize(floor(raw.size() / 2.0)+1);

		//nc::NdArray<double> phase = nc::NdArray<double>(get_angle(raw));
		fft = nc::NdArray<double>(get_abs(raw));
		/*freqs = nc::arange<double>(ceil(L / 2.)+2) * (fps / L);*/
		freqs = nc::arange<double>(ceil(L / 2.)) * (fps / L);

		nc::NdArray<double> freqs_tmp = freqs * 60.;
		nc::NdArray<bool> idx = nc::where((freqs_tmp > (double)63 & freqs_tmp < (double)180), nc::NdArray<bool>(freqs_tmp.shape()).fill(true), nc::NdArray<bool>(freqs_tmp.shape()).fill(false));
		//vector<double> proper_fft;
		//int num = 0;
		//auto fft_it = fft.begin();
		//for (auto it = idx.begin(); it < idx.end(); ++it)
		//{
		//	if (*it)
		//	{
		//		proper_fft.push_back(*(fft_it+num));
		//		num++;
		//	}
		//	//logfileDetailed << *it << ",";
		//}
		//nc::NdArray<double> pruned = nc::NdArray<double>(proper_fft);
		
		nc::NdArray<double> pruned = fft.getByMask(idx);

		nc::NdArray<double> pfreq = freqs_tmp.getByMask(idx);

		nc::NdArray<nc::uint32> idx2 = nc::argmax(pruned);
		bpm = *pfreq[idx2].data();
		
		if (output_bpms.rows >= 48) {
			delete_rows(output_bpms);
		}

		if (output_bpms.rows == 0)
		{
			output_bpms.push_back(bpm);
		}
		else
		{
			output_bpms.push_back(bpm);
		}

	// 以下21行是为了调整而将每帧参数输出到CSV文件中的代码，可以直接注释
		if (bpms.rows > 0)
		{
			logfileDetailed << s.at<double>(s.rows - 1) << ",";
			logfileDetailed << fps << ",";
			logfileDetailed << bpm << ",";
			logfileDetailed << ",";
			for (auto it = pfreq.begin(); it < pfreq.end(); ++it)
			{
				logfileDetailed << *it << ",";
			}
			for (auto it = pruned.begin(); it < pruned.end(); ++it)
			{
				logfile << *it << ",";
			}
		}

	}
	logfileDetailed << "\n";
	logfile << "\n";
	logfile.flush();
	logfileDetailed.flush();
}

void RPPG::log() {
	//// 记录平均心率
	//logfile << framecount << ",";
	//logfile << time << ",";
	//logfile << meanBpm << "\n";
	//logfile.flush();

	//// 记录每帧的计算的心率以及辅助信息
	//logfileDetailed << framecount << ",";
	//logfileDetailed << time << ",";
	//if (bpms.rows > 0)
	//{
	//	logfileDetailed << bpms.at<double>(bpms.rows - 1) << ",";
	//	logfileDetailed <<  ",";
	//	logfileDetailed << s.at<double>(s.rows - 1) << "\n";
	//}
	//else
	//{
	//	logfileDetailed << 0 << ",";
	//	logfileDetailed << 0 << "\n";
	//}
	//
	//logfileDetailed.flush();

}


void RPPG::draw(cv::Mat &frameRGB) {

    // 画人的额头框
	rectangle(frameRGB, roi, RED);

    // 画人脸框
    rectangle(frameRGB, box, RED);
	
    std::stringstream ss;

    // Draw BPM text
    if (faceValid) {

		ss.str("");
		ss << "bpm: ";
		ss.precision(3);
		ss << meanBpm;
		putText(frameRGB, ss.str(), Point(box.tl().x, box.tl().y - 10), FONT_HERSHEY_PLAIN, 2, GREEN, 2);
    }
	
	//// 写出瞬时心率
	//if (bpms.rows > 1)
	//{
	//	ss.str("");
	//	ss << " forehead";
	//	ss.precision(3);
	//	ss << bpms.at<double>(bpms.rows - 1);
	//	putText(frameRGB, ss.str(), Point(1, 50), FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 1.5);
	//}

	// 在图上写出当前为第几帧
	//ss.str("");
	//ss << framecount;
	//putText(frameRGB, ss.str(), Point(1, 470), FONT_HERSHEY_SIMPLEX, 0.8, RED , 1.5);
	
}
