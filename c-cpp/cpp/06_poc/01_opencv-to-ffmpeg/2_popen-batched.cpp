#include "common.h"   

using namespace std;
using namespace cv;                                                          

int main() {
    VideoCapture cap = openVideoSource();

    FILE *output;
    output = popen(ffmpegCommand.c_str(), "w");

    size_t count = 0;
    Mat frame;
    while (cap.read(frame)) {                                                       
        fwrite(frame.data, 1, frame.dataend - frame.datastart, output);
        if ((count++) > 1024) {
            cout << "Length limited reached, exiting..." << endl;
            break;
        }
        
    }  
    pclose(output);                                                                      
    return 0;                                                                
} 
