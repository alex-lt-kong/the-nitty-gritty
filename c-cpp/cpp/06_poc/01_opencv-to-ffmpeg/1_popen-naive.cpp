#include "common.h"

using namespace std;
using namespace cv;

int main() {
    FILE *output;
    output = popen(ffmpegCommand.c_str(), "w");
    VideoCapture cap = openVideoSource();
    Mat frame;
    size_t count = 0;
    while (cap.read(frame)) {
        for (int64_t i = 0; i < frame.dataend - frame.datastart; ++i) { 
           fwrite(&frame.data[i], sizeof(frame.data[i]), 1, output);
        }
        if ((count++) > 1024) {
            cout << "Length limited reached, exiting..." << endl;
            break;
        }
    }  
    pclose(output);
    return 0;
} 
