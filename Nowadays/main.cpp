#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String banana_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier banana_cascade;
String window_name = "Capture - Face detection";

/** @function main */
int main(void)
{
	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!banana_cascade.load(banana_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };

	//-- 2. Read the video stream
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);

		int c = waitKey(10);
		if ((char)c == 27) { break; } // escape
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> bananas;
	Mat frame_gray;

	//Conversion of frame to grayscale
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	//Contrast enhance(Spread out intensity distribution)
	equalizeHist(frame_gray, frame_gray);
	//-- Detect Banana
	banana_cascade.detectMultiScale(frame_gray, bananas, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	cout << bananas.size();

	for (size_t i = 0; i < bananas.size(); i++)
	{
		Point center(bananas[i].x + bananas[i].width / 2, bananas[i].y + bananas[i].height / 2);
		ellipse(frame, center, Size(bananas[i].width / 2, bananas[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

	}
	//-- Show what you got
	imshow(window_name, frame);
}