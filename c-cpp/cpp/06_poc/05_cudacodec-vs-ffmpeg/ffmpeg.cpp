#include "utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char *argv[]) {
  cout << getBuildInformation() << endl;
  if (argc != 3) {
    cerr << "Usage : " << argv[0] << "  <Source URI> <Dest path>" << endl;
    return -1;
  }

  install_signal_handler();
  cout << "A signal handler is installed, "
          "press Ctrl+C to exit the event loop gracefully."
       << endl;

  VideoCapture cap;
  bool result = cap.open(string(argv[1]), CAP_ANY,
                         {CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY});
  if (!result) {
    cerr << "Error!" << endl;
  }
  VideoWriter vwriter = VideoWriter(
      string(argv[2]), CAP_ANY, VideoWriter::fourcc('a', 'v', 'c', '1'), 25.0,
      Size(1280, 720),
      {VIDEOWRITER_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY});

  Mat hFrameCurr, hFramePrev;
  size_t frameCount = 0;
  while (!e_flag) {
    if (!cap.read(hFrameCurr)) {
      cerr << "cap.read(hFrame) is False" << endl;
      this_thread::sleep_for(10000ms);
      cap.open(string(argv[1]), CAP_ANY,
               {CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY});
    }
    ++frameCount;
    auto t1 = chrono::high_resolution_clock::now();
    float diff = getFrameChanges(hFramePrev, hFrameCurr);
    auto t2 = chrono::high_resolution_clock::now();

    if (!hFrameCurr.empty()) {
      hFramePrev = hFrameCurr.clone();
      overlayDatetime(hFrameCurr);
      // rotate(hFrameCurr, hFrameCurr, ROTATE_180);
      vwriter.write(hFrameCurr);
    } else {
      cerr << "frameCount: " << frameCount << " is empty" << endl;
    }
    auto t3 = chrono::high_resolution_clock::now();
    if (!hFramePrev.empty() && frameCount % 100 == 0) {
      cout << "frameCount: " << frameCount << ", size(): " << hFrameCurr.size()
           << ", channels(): " << hFrameCurr.channels() << ", diff: " << fixed
           << setprecision(2) << diff << "% ("
           << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()
           << " ms) , iteration took: "
           << chrono::duration_cast<chrono::milliseconds>(t3 - t1).count()
           << " ms" << endl;
    }
  }
  vwriter.release();
  cout << "vwriter.release()ed" << endl;
  return 0;
}
