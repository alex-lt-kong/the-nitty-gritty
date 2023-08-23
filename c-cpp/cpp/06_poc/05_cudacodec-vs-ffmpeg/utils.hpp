#include "opencv2/core/cuda.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

static volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signum) {
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + (char)(signum / 10);
  msg[9] = '0' + (char)(signum % 10);
  (void)write(STDIN_FILENO, msg, strlen(msg));
  e_flag = 1;
}

inline void install_signal_handler() {
  // This design canNOT handle more than 99 signal types
  if (_NSIG > 99) {
    fprintf(stderr, "signal_handler() can't handle more than 99 signals\n");
    abort();
  }
  struct sigaction act;
  // Initialize the signal set to empty, similar to memset(0)
  if (sigemptyset(&act.sa_mask) == -1) {
    perror("sigemptyset()");
    abort();
  }
  act.sa_handler = signal_handler;
  /* SA_RESETHAND means we want our signal_handler() to intercept the signal
  once. If a signal is sent twice, the default signal handler will be used
  again. `man sigaction` describes more possible sa_flags. */
  act.sa_flags = SA_RESETHAND;
  // act.sa_flags = 0;
  if (sigaction(SIGINT, &act, 0) == -1) {
    perror("sigaction()");
    abort();
  }
}

inline float getCudaFrameChanges(cudacodec::GpuMat &prevFrame,
                                 cudacodec::GpuMat &currFrame,
                                 cudacodec::GpuMat &diffFrame) {
  if (prevFrame.empty() || currFrame.empty()) {
    return -1;
  }
  if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) {
    return -1;
  }
  if (prevFrame.cols == 0 || prevFrame.rows == 0) {
    return -1;
  }

  cuda::absdiff(prevFrame, currFrame, diffFrame);
  cuda::cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
  cuda::threshold(diffFrame, diffFrame, 32, 255, THRESH_BINARY);
  int nonZeroPixels = cuda::countNonZero(diffFrame);
  return 100.0 * nonZeroPixels / (diffFrame.rows * diffFrame.cols);
}

inline float getFrameChanges(Mat &prevFrame, Mat &currFrame) {
  Mat diffFrame;
  if (prevFrame.empty() || currFrame.empty()) {
    return -1;
  }
  if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) {
    return -1;
  }
  if (prevFrame.cols == 0 || prevFrame.rows == 0) {
    return -1;
  }

  absdiff(prevFrame, currFrame, diffFrame);
  cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
  threshold(diffFrame, diffFrame, 32, 255, THRESH_BINARY);
  int nonZeroPixels = countNonZero(diffFrame);
  return 100.0 * nonZeroPixels / (diffFrame.rows * diffFrame.cols);
}

inline string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t current_time = std::chrono::system_clock::to_time_t(now);
  std::tm *time_info = std::localtime(&current_time);
  std::ostringstream oss;
  oss << std::put_time(time_info, "%Y-%m-%dT%H:%M:%S");
  // cout << oss.str() << endl;
  string dt = oss.str();
  return dt;
}

inline void overlayDatetime(Mat &frame) {
  time_t now;
  time(&now);
  // char buf[sizeof "1970-01-01 00:00:00"];
  // strftime(buf, sizeof buf, "%F %T", localtime(&now));
  string ts = getCurrentTimestamp();

  Size textSize = getTextSize(ts, FONT_HERSHEY_DUPLEX, 1, 8 * 1, nullptr);
  putText(frame, ts, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, 1,
          Scalar(0, 0, 0), 8 * 1, LINE_8, false);
  putText(frame, ts, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, 1,
          Scalar(255, 255, 255), 2 * 1, LINE_8, false);
  /*
  void putText 	(InputOutputArray  	img,
                      const String &  	text,
                      Point  	org,
                      int  	fontFace,
                      double  	textOverlayFontSacle,
                      Scalar  	color,
                      int  	thickness = 1,
                      int  	lineType = LINE_8,
                      bool  	bottomLeftOrigin = false
      )
  */
}