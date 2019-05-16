#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvstd.hpp>
#include <cairo/cairo.h>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace cv;
using namespace std;
#define res 8
#define printAlphabet false

void putTextCairo(
	cv::Mat& targetImage,
	std::string const& text,
	cv::Point2d centerPoint,
	std::string const& fontFace,
	double fontSize,
	cv::Scalar textColor,
	bool fontItalic,
	bool fontBold);

//To Do:
//1.)Move code from main into functions(maybe inline)
//2.)introduce OpenMP to the code

void constructAlphabet(Mat* imgs, String* alphabet, String const fontFace, double fontSize, Scalar textColor, Scalar backColor, int size);

Mat* approximateBlock(Mat* src, Mat alphabet[], int alength, int x, int y, int blocksize) {
	int closest = 0;
	double optrating = DBL_MAX;
	for (int k = 0; k < alength; k++) {
		double currrating = 0.0;
		//uint8_t* currpixelPtr = (uint8_t*)alph[k].data;
		//int currcn = alph[i].channels();
		for (int i = x; i < x + blocksize; i++) {
			for (int j = y; j < y + blocksize; j++) {
				//currrating += abs(pixelPtr[x * grayImg.cols + y + 0] - currpixelPtr[(x - i) * alph[k].cols + (y - j) + 0]);
				currrating += pow(abs((*src).at<uchar>(i, j) - alphabet[k].at<uchar>(i - x, j - y)), 16);
				int dbg = 0;
			}
			int dbg = 0;
		}
		if (k == alength - 2) {
			int dbg = 0;
		}
		if (currrating < optrating) {
			optrating = currrating;
			closest = k;
		}
	}
	return &alphabet[closest];
}

int main(int argc, char** argv)
{	
	//Try to load the image
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat grayImg, approx;

	cvtColor(image, grayImg, COLOR_BGR2GRAY);

	int xchunks, ychunks;

	if (grayImg.rows % res != 0 || grayImg.cols % res != 0) {
		cout << "Image dimensions are not a multiple of the specified resolution." << endl;
		return -1;
	}

	xchunks = (int)(grayImg.rows/res);
	ychunks = (int)(grayImg.cols/res);
	if (xchunks * res != grayImg.rows || ychunks * res != grayImg.cols) {
		cout << "wat." << endl;
	}

	
	approx = Mat(grayImg.rows, grayImg.cols, CV_8UC1, Scalar(255));  //Intialise Target image

	String alphabetStr = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
	String alphabet[94];
	for (int i = 0; i < 94; i++) {
		alphabet[i] = alphabetStr[i];
	}

	//create the alphabet chunks
	Mat alph[101];

	constructAlphabet(alph, alphabet, "arial", res, Scalar(0), Scalar(255), 94);
	
	//for white, gray and black blocks
	
	alph[94] = Mat(res, res, CV_8UC1, Scalar(0));
	alph[95] = Mat(res, res, CV_8UC1, Scalar(255));
	alph[96] = Mat(res, res, CV_8UC1, Scalar(63));
	alph[97] = Mat(res, res, CV_8UC1, Scalar(127));
	alph[98] = Mat(res, res, CV_8UC1, Scalar(191));
	alph[99] = Mat(res, res, CV_8UC1, Scalar(85));
	alph[100] = Mat(res, res, CV_8UC1, Scalar(171));

	//approximate the source image
	
	for (int i = 0; i < grayImg.rows; i+=res) {
		#pragma omp parallel for 
		for (int j = 0; j < grayImg.cols; j+=res) {
			Mat* ablock = approximateBlock(&grayImg, alph, 94, i, j, res);
			for (int x = i; x < i + res; x++) {
				for (int y = j; y < j + res; y++) {
					approx.at<uchar>(x, y) = (*ablock).at<uchar>(x - i, y - j);
				}
			}
		}
	}
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);
	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", grayImg);
	namedWindow("Grayscale approximation", WINDOW_AUTOSIZE);
	imshow("Grayscale approximation", approx);
	imwrite("approx.bmp", approx);
	imwrite("gray.bmp", grayImg);
	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}

void constructAlphabet(Mat* imgs, String* alphabet, String const fontFace, double fontSize, Scalar textColor, Scalar backColor, int size){
	for (int i = 0; i < size; i++) {
		imgs[i] = Mat(res, res, CV_8UC1, backColor);
		putTextCairo(imgs[i], alphabet[i], cvPoint((int)(res / 2), (int)(res / 2)), fontFace, fontSize, textColor, false, true);
		if (printAlphabet) {
			imwrite(alphabet[i] + ".bmp", imgs[i]);
		}
	}
}


void approximateImage(Mat* src, Mat* targt, Mat alphabet[]){
	return;
}


//putTexCairo function taken from stackoverflow
//https://stackoverflow.com/questions/11917124/opencv-how-to-use-other-font-than-hershey-with-cvputtext-like-arial/26307882

void putTextCairo(
	cv::Mat& targetImage,
	std::string const& text,
	cv::Point2d centerPoint,
	std::string const& fontFace,
	double fontSize,
	cv::Scalar textColor,
	bool fontItalic,
	bool fontBold)
{
	// Create Cairo
	cairo_surface_t* surface =
		cairo_image_surface_create(
			CAIRO_FORMAT_ARGB32,
			targetImage.cols,
			targetImage.rows);

	cairo_t* cairo = cairo_create(surface);

	// Wrap Cairo with a Mat
	cv::Mat cairoTarget(
		cairo_image_surface_get_height(surface),
		cairo_image_surface_get_width(surface),
		CV_8UC4,
		cairo_image_surface_get_data(surface),
		cairo_image_surface_get_stride(surface));

	// Put image onto Cairo
	//cv::cvtColor(targetImage, cairoTarget, cv::COLOR_BGR2BGRA);
	
	cvtColor(targetImage, cairoTarget, COLOR_GRAY2BGRA);
	// Set font and write text
	cairo_select_font_face(
		cairo,
		fontFace.c_str(),
		fontItalic ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL,
		fontBold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);

	cairo_set_font_size(cairo, fontSize);
	cairo_set_source_rgb(cairo, textColor[2], textColor[1], textColor[0]);

	cairo_text_extents_t extents;
	cairo_text_extents(cairo, text.c_str(), &extents);

	cairo_move_to(
		cairo,
		centerPoint.x - extents.width / 2 - extents.x_bearing,
		centerPoint.y - extents.height / 2 - extents.y_bearing);
	cairo_show_text(cairo, text.c_str());

	// Copy the data to the output image
	cv::cvtColor(cairoTarget, targetImage, cv::COLOR_BGRA2GRAY);

	cairo_destroy(cairo);
	cairo_surface_destroy(surface);
}