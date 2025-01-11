#include "OpenCLUtils.h"
#include "OpenCVUtils.h"

#include "opencv2/opencv.hpp"

#include "Cl/cl.h"

#include <vector>
#include <string>

cl_device_id device = nullptr;
cl_context context = nullptr;
cl_program program = nullptr;
cl_kernel kernel = nullptr;
cl_command_queue queue = nullptr;
cl_int err = -1;

bool EdgeDetect(const cv::Mat& input,
				float dot_radius,
				float scale,
                cv::Mat& output)
{
	const size_t inputDataSize = input.cols * input.rows * input.channels() * sizeof(unsigned char);
	const size_t outputDataSize = input.cols * input.rows * output.channels() * sizeof(unsigned char);
	cl_mem inputA = OpenCLUtils::create_input_buffer(context, input.data, inputDataSize);
	cl_mem output_buffer = OpenCLUtils::create_output_buffer(context, outputDataSize);

	const int width = input.cols;
	const int height = input.rows;
	const int channels = input.channels();

    /* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(float), &dot_radius);
	err |= clSetKernelArg(kernel, 2, sizeof(float), &scale);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &width);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_buffer);
    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        return false;
    }

	size_t global[2] = { width, height };

    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 2,
                                 NULL,
                                 (const size_t*)&global,
                                 NULL,
                                 0,
                                 NULL,
                                 NULL);

    if (err < 0)
    {
        perror("Couldn't enqueue the kernel");
        return false;
    }

    /* Read the kernel's output    */
    err = clEnqueueReadBuffer(queue,
                              output_buffer,
                              CL_TRUE,
                              0,
                              outputDataSize,
                              output.data,
                              0,
                              NULL,
                              NULL);
    if (err < 0)
    {
        perror("Couldn't read the buffer");
        return false;
    }

	clReleaseKernel(kernel);
	clReleaseMemObject(inputA);
	clReleaseMemObject(output_buffer);
    return true;
}

bool InitializeDeviceAndContext()
{
	device = OpenCLUtils::create_device();
	if (!device)
	{
		return false;
	}

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		return false;
	}
    return true;
}

bool InitializeProgram()
{
	/* Build program */
	program = OpenCLUtils::build_program(context, device, "shaders/halftoning_img.cl");
	if (!program)
		return false;

	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		return false;
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, "halftone", &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		return false;
	};
    return true;
}

int main()
{
	const float dot_radius = 5;
	const float scale = 10;

	if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;

	cv::Mat inputImg = cv::imread("content/test.jpg");
	if (!OpenCVUtils::ConvertType(inputImg, CV_8UC4))
	{
		printf("Failed to Convert Output Image Type!");
		return -1;
	}

	cv::Mat outputImg(inputImg.rows, inputImg.cols, inputImg.type(), cv::Scalar(0, 0, 0));

	cv::Mat inputImgRGBA;
	cv::cvtColor(inputImg, inputImgRGBA, cv::COLOR_BGRA2RGBA);
    if (!EdgeDetect(inputImgRGBA, dot_radius, scale, outputImg))
        return -1;

    /// Check Results ---------------------------------------------------------

	const size_t combinedWidth = inputImg.cols + outputImg.cols;
	cv::Mat combinedImage(cv::Size(combinedWidth, inputImg.rows), inputImg.type());

	// Copy the first image into the left side of the combined image
    inputImg.copyTo(combinedImage(cv::Rect(0, 0, inputImg.cols, inputImg.rows)));

	// Copy the second image into the right side of the combined image
	cv::cvtColor(outputImg, outputImg, cv::COLOR_RGBA2BGRA);
	outputImg.copyTo(combinedImage(cv::Rect(inputImg.cols, 0, outputImg.cols, outputImg.rows)));
    
	cv::imshow("Side by Side Result", combinedImage);
	cv::waitKey();
    
    ///* Deallocate resources */
    
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}