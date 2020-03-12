//
//  Heartbeat.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 4/06/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "Heartbeat.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv.hpp"
#include "Baseline.hpp"
#include <time.h>
#include <opencv2/videoio/videoio_c.h>

#define DEFAULT_RPPG_ALGORITHM "g"
#define DEFAULT_FACEDET_ALGORITHM "haar"
#define DEFAULT_RESCAN_FREQUENCY 1
#define DEFAULT_SAMPLING_FREQUENCY 1
#define DEFAULT_MIN_SIGNAL_SIZE 5
#define DEFAULT_MAX_SIGNAL_SIZE 5
#define DEFAULT_DOWNSAMPLE 1 // x means only every xth frame is used

#define HAAR_CLASSIFIER_PATH "haarcascade_frontalface_alt.xml"
#define DNN_PROTO_PATH "opencv/deploy.prototxt"
#define DNN_MODEL_PATH "opencv/res10_300x300_ssd_iter_140000.caffemodel"

using namespace cv;

Heartbeat::Heartbeat(int argc_, char * argv_[], bool switches_on_) {

    argc = argc_;
    argv.resize(argc);
    copy(argv_, argv_ + argc, argv.begin());
    switches_on = switches_on_;

    // map the switches to the actual
    // arguments if necessary
    if (switches_on) {

        vector<string>::iterator it1, it2;
        it1 = argv.begin();
        it2 = it1 + 1;

        while (true) {

            if (it1 == argv.end()) break;
            if (it2 == argv.end()) break;

            if ((*it1)[0] == '-')
                switch_map[*it1] = *(it2);

            it1++;
            it2++;
        }
    }
}

string Heartbeat::get_arg(int i) {

    if (i >= 0 && i < argc)
        return argv[i];

    return "";
}

string Heartbeat::get_arg(string s) {

    if (!switches_on) return "";

    if (switch_map.find(s) != switch_map.end())
        return switch_map[s];

    return "";
}

bool to_bool(string s) {
    bool result;
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    istringstream is(s);
    is >> boolalpha >> result;
    return result;
}

rPPGAlgorithm to_rppgAlgorithm(string s) {
    rPPGAlgorithm result;
    if (s == "g") result = g;
    else if (s == "pca") result = pca;
    else if (s == "xminay") result = xminay;
    else {
        std::cout << "Please specify valid rPPG algorithm (g, pca, xminay)!" << std::endl;
        exit(0);
    }
    return result;
}

faceDetAlgorithm to_faceDetAlgorithm(string s) {
    faceDetAlgorithm result;
	if (s == "haar") result = haar;
	else if (s == "deep") result = deep;
	else if (s == "mtcnn_deep") result = mtcnn_deep;
    else {
        std::cout << "Please specify valid face detection algorithm (haar, deep)!" << std::endl;
        exit(0);
    }
    return result;
}

int main(int argc, char * argv[]) {

    Heartbeat cmd_line(argc, argv, true);

    string input = cmd_line.get_arg("-i"); // Filepath for offline mode
    std::cout << "input = " + input << std::endl;

    // algorithm setting
    rPPGAlgorithm rPPGAlg;
    string rppgAlgString = cmd_line.get_arg("-rppg");
    if (rppgAlgString != "") {
        rPPGAlg = to_rppgAlgorithm(rppgAlgString);
    } else {
        rPPGAlg = to_rppgAlgorithm(DEFAULT_RPPG_ALGORITHM);
    }

    cout << "Using rPPG algorithm " << rPPGAlg << "." << endl;

    // face detection algorithm setting
    faceDetAlgorithm faceDetAlg;
    // string faceDetAlgString = cmd_line.get_arg("-facedet");
	string faceDetAlgString = "mtcnn_deep";
    if (faceDetAlgString != "") {
        faceDetAlg = to_faceDetAlgorithm(faceDetAlgString);
    } else {
        faceDetAlg = to_faceDetAlgorithm(DEFAULT_FACEDET_ALGORITHM);
    }

    cout << "Using face detection algorithm " << faceDetAlg << "." << endl;

    // rescanFrequency setting
    double rescanFrequency;
    string rescanFrequencyString = cmd_line.get_arg("-r");
    if (rescanFrequencyString != "") {
        rescanFrequency = atof(rescanFrequencyString.c_str());
    } else {
        rescanFrequency = DEFAULT_RESCAN_FREQUENCY;
    }

    // samplingFrequency setting
    double samplingFrequency;
    string samplingFrequencyString = cmd_line.get_arg("-f").c_str();
    if (samplingFrequencyString != "") {
        samplingFrequency = atof(samplingFrequencyString.c_str());
    } else {
        samplingFrequency = DEFAULT_SAMPLING_FREQUENCY;
    }

    // max signal size setting
    int maxSignalSize;
    string maxSignalSizeString = cmd_line.get_arg("-max");
    if (maxSignalSizeString != "") {
        maxSignalSize = atof(maxSignalSizeString.c_str());
    } else {
        maxSignalSize = DEFAULT_MAX_SIGNAL_SIZE;
    }

    // min signal size setting
    int minSignalSize;
    string minSignalSizeString = cmd_line.get_arg("-min");
    if (minSignalSizeString != "") {
        minSignalSize = atof(minSignalSizeString.c_str());
    } else {
        minSignalSize = DEFAULT_MIN_SIGNAL_SIZE;
    }

    // visualize baseline setting
    string baseline_input = cmd_line.get_arg("-baseline");

    if (minSignalSize > maxSignalSize) {
        std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
        exit(0);
    }

    // Reading gui setting
    bool gui = false;

    string guiString = cmd_line.get_arg("-gui");
    if (guiString != "") {
        gui = to_bool(guiString);
    } else {
        gui = true;
    }

    // Reading log setting
    bool log;
    string logString = cmd_line.get_arg("-log");
    if (logString != "") {
        log = to_bool(logString);
    } else {
        log = true;
    }

    // Reading downsample setting
    int downsample;
    string downsampleString = cmd_line.get_arg("-ds");
    if (downsampleString != "") {
        downsample = atof(downsampleString.c_str());
    } else {
        downsample = DEFAULT_DOWNSAMPLE;
    }

    bool offlineMode = input != "";

    std::cout << "start open data"<< std::endl;
    VideoCapture cap;
    if (offlineMode) {
         cap.open(input);
        //cap = VideoCapture(input);
        std::cout << "offlineMode cap.open(" + input + ")"<< std::endl;
    } else {
        //cap.open(0);
        cap.open(1);
        std::cout << "onlineMode cap.open(0)"<< std::endl;
    }
    if (!cap.isOpened()) {
        std::cout << "cap.open failed, exit!" << std::endl;
        return -1;
    }

	//VideoCapture videoReader("outdoor_90bpm.avi");

    std::string title = offlineMode ? "rPPG offline" : "rPPG online";
    cout << title << endl;
    cout << "Processing " << (offlineMode ? input : "live feed") << endl;

    // Configure logfile path
    string LOG_PATH;
    if (offlineMode) {
        LOG_PATH = input.substr(0, input.find_last_of("."));
    } else {
        std::ostringstream filepath;
        filepath << "Live_ffmpeg";
        LOG_PATH = filepath.str();
    }

    // Load video information
    const int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double FPS = cap.get(cv::CAP_PROP_FPS);
    const double TIME_BASE = 0.001;

    // Print video information
    cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
    cout << "FPS: " << FPS << endl;
    cout << "TIME BASE: " << TIME_BASE << endl;

    std::ostringstream window_title;
    window_title << title << " - " << WIDTH << "x" << HEIGHT << " -rppg " << rPPGAlg << " -facedet " << faceDetAlg << " -r " << rescanFrequency << " -f " << samplingFrequency << " -min " << minSignalSize << " -max " << maxSignalSize << " -ds " << downsample;

    // Set up rPPG
    RPPG rppg = RPPG();
    rppg.load(rPPGAlg, faceDetAlg,
              WIDTH, HEIGHT, TIME_BASE, downsample,
              samplingFrequency, rescanFrequency,
              minSignalSize, maxSignalSize,
              LOG_PATH, HAAR_CLASSIFIER_PATH,
              DNN_PROTO_PATH, DNN_MODEL_PATH,
              log, gui, 250);

    // Load baseline if necessary
    Baseline baseline = Baseline();
    if (baseline_input != "") {
        baseline.load(1, 0.000001, baseline_input);
    }

    cout << "START ALGORITHM" << endl;

    int i = 0;
    Mat frameRGB, frameGray;

	// 用于保存处理结果
	String outFile = "result.avi";
	Size frameSize(640, 480);
	VideoWriter oVideoWriter(outFile, CV_FOURCC('D', 'I', 'V', 'X'), 20, frameSize, true);

    while (true) {

		int start = (cv::getTickCount() * 1000.0) / cv::getTickFrequency(); // 记录一帧开始时间，用于测试运行时间

        if (offlineMode) {
            // 从视频中读取视频
            //videoReader >> frameRGB;
            cap >> frameRGB;
        } else {
            // 从摄像头读取视频
            cap.read(frameRGB);
        }

		if (frameRGB.empty())
            break;

		// 计算当前是第几帧
		rppg.countFrame();

		// 对图像做镜像翻转
		flip(frameRGB, frameRGB, 1);

        // 产生灰度图像
        cvtColor(frameRGB, frameGray, COLOR_BGR2GRAY);
        equalizeHist(frameGray, frameGray);

        int time;
		// 如果读取视频文件进行测试，建议打开下行注释
		// offlineMode = true;
		if (offlineMode) {
		    //time = videoReader.get(cv::CAP_PROP_POS_MSEC);
            time = cap.get(cv::CAP_PROP_POS_MSEC);
		} else {
		    time = (cv::getTickCount()*1000.0)/cv::getTickFrequency();
		}

		// 主要处理部分
        if (i % downsample == 0) {
			rppg.processFrame(frameRGB, frameGray, time, rppg.face_detector);
        } else {
            cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
        }

        if (baseline_input != "") {
            baseline.processFrame(frameRGB, time);
        }

		int after_time = (cv::getTickCount() * 1000.0) / cv::getTickFrequency() - start;

		// count time and draw on pics.
		std::stringstream my_fps;
		my_fps.str("");
		my_fps << "time: " << after_time;
		putText(frameRGB, my_fps.str(), Point(1, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 2);

        oVideoWriter << frameRGB;
        if (gui) {
            imshow(window_title.str(), frameRGB);
			oVideoWriter << frameRGB;
           if (waitKey(30) >= 0) break;
        }
        
        i++;
    }
	oVideoWriter.release();
    if (offlineMode) {
        //videoReader.release();
        cap.release();
    }

    return 0;
}
