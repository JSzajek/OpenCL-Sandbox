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

bool GrayscaleImage(const cv::Mat& input,
                    cv::Mat& output)
{
	const size_t inputDataSize = input.cols * input.rows * input.channels() * sizeof(unsigned char);
	const size_t outputDataSize = input.cols * input.rows * output.channels() * sizeof(unsigned char);
	cl_mem inputA = OpenCLUtils::create_input_buffer(context, input.data, inputDataSize);
	cl_mem output_buffer = OpenCLUtils::create_output_buffer(context, outputDataSize);

	const int width = input.cols;
	const int height = input.rows;

    /* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
	err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_buffer);
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
	program = OpenCLUtils::build_program(context, device, "shaders/grayscale.cl");
	if (!program)
		return false;

	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		return false;
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, "grayscale", &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		return false;
	};
    return true;
}

int main() 
{
	if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;

	cv::Mat inputImg = cv::imread("content/test.jpg");
	cv::Mat outputImg(inputImg.rows, inputImg.cols, CV_8UC1, cv::Scalar(0, 0, 0));

    if (!GrayscaleImage(inputImg, outputImg))
        return -1;

    /// Check Results ---------------------------------------------------------

	const size_t combinedWidth = inputImg.cols + outputImg.cols;
	cv::Mat combinedImage(cv::Size(combinedWidth, inputImg.rows), inputImg.type());

	// Copy the first image into the left side of the combined image
    inputImg.copyTo(combinedImage(cv::Rect(0, 0, inputImg.cols, inputImg.rows)));

    if (!OpenCVUtils::ConvertType(outputImg, inputImg.type()))
    {
        printf("Failed to Convert Output Image Type!");
        return -1;
    }

	// Copy the second image into the right side of the combined image
    outputImg.copyTo(combinedImage(cv::Rect(inputImg.cols, 0, outputImg.cols, outputImg.rows)));
    
	cv::imshow("Side by Side Result", combinedImage);
	cv::waitKey();
    
    ///* Deallocate resources */
    
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}