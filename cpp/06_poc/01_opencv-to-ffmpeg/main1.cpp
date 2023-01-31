#include <iostream>                                                                                                    
#include <vector>                        
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"              

using namespace std;
using namespace cv;                                                          

int main() {   
    
    bool result;
    VideoCapture cap = VideoCapture();

    result = cap.open("/dev/video0");
    if (result == false) { cout << "Failed to open, returned." << endl; return 1;}
    
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    FILE *output;
    output = popen ("/usr/bin/ffmpeg -y -f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 30 -i pipe:0 ./main1.mp4 2> ./main1.log", "w");
    Mat frame;                                                               
    
    int count = 0;
    while (cap.read(frame)) {                                                       
        fwrite(frame.data, 1, frame.dataend - frame.datastart, output);       
        count ++;
        if (count > 300) { break; }                                   
    }  
    pclose(output);                                                                      
    return 0;                                                                
} 
