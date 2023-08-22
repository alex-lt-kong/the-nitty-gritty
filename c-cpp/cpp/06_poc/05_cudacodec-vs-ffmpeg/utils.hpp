#include "opencv2/core/cuda.hpp"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

inline float getCudaFrameChanges(cv::cudacodec::GpuMat &prevFrame,
                                 cv::cudacodec::GpuMat &currFrame) {
  cv::cudacodec::GpuMat diffFrame;
  if (prevFrame.empty() || currFrame.empty()) {
    return -1;
  }
  if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) {
    return -1;
  }
  if (prevFrame.cols == 0 || prevFrame.rows == 0) {
    return -1;
  }

  cv::cuda::absdiff(prevFrame, currFrame, diffFrame);
  cv::cuda::cvtColor(diffFrame, diffFrame, cv::COLOR_BGR2GRAY);
  cv::cuda::threshold(diffFrame, diffFrame, 1, 255, cv::THRESH_BINARY);
  int nonZeroPixels = cv::cuda::countNonZero(diffFrame);
  return 100.0 * nonZeroPixels / (diffFrame.rows * diffFrame.cols);
}

inline float getFrameChanges(cv::Mat &prevFrame, cv::Mat &currFrame) {
  cv::Mat diffFrame;
  if (prevFrame.empty() || currFrame.empty()) {
    return -1;
  }
  if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) {
    return -1;
  }
  if (prevFrame.cols == 0 || prevFrame.rows == 0) {
    return -1;
  }

  cv::absdiff(prevFrame, currFrame, diffFrame);
  cv::cvtColor(diffFrame, diffFrame, cv::COLOR_BGRA2GRAY);
  cv::threshold(diffFrame, diffFrame, 1, 255, cv::THRESH_BINARY);
  int nonZeroPixels = cv::countNonZero(diffFrame);
  return 100.0 * nonZeroPixels / (diffFrame.rows * diffFrame.cols);
}