#include <csignal>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signum) {
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + (char)(signum / 10);
  msg[9] = '0' + (char)(signum % 10);
  (void)write(STDIN_FILENO, msg, strlen(msg));
  e_flag = 1;
}

void install_signal_handler() {
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

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage : " << argv[0] << "  <Source URI> <Dest path>"
              << std::endl;
    return -1;
  }
  std::cout << cv::getBuildInformation() << std::endl;

  install_signal_handler();
  std::cout << "A signal handler is installed, "
               "press Ctrl+C to exit the event loop gracefully."
            << std::endl;

  cv::cuda::GpuMat dFrame;

  cv::Ptr<cv::cudacodec::VideoReader> dReader =
      cv::cudacodec::createVideoReader(std::string(argv[1]));
  cv::Ptr<cv::cudacodec::VideoWriter> dWriter =
      cv::cudacodec::createVideoWriter(
          std::string(argv[2]), cv::Size(1280, 720), cv::cudacodec::Codec::H264,
          25.0, cv::cudacodec::ColorFormat::BGRA);
  size_t frameCount = 0;
  while (!e_flag) {

    if (!dReader->nextFrame(dFrame)) {
      std::cerr << "dReader->nextFrame(dFrame) is False" << std::endl;
      break;
    }

    std::cout << "frameCount: " << ++frameCount << ", size(): " << dFrame.size()
              << ", channels(): " << dFrame.channels() << std::endl;
    if (!dFrame.empty()) {
      dWriter->write(dFrame);
    }
  }
  dWriter->release();
  std::cout << "dWriter->release()ed" << std::endl;
  return 0;
}
