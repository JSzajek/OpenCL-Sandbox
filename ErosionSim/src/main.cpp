#include "OpenCLUtils.h"
#include "OpenCVUtils.h"
#include "RandomUtils.h"
#include "Timer.h"

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
	program = OpenCLUtils::build_program(context, device, "shaders/erosion.cl");
	if (!program)
		return false;

	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		return false;
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, "simulate", &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		return false;
	};
    return true;
}

int main()
{
	const float ErosionRate = 0.005f;
	const float DepositionRate = 0.01f;
	const float FlowSpeed = 1.0f;

	const uint8_t WaterThresholdValue = 128;

	// Create float-based heightmap and water map
	cv::Mat inputImg = cv::imread("content/test_map.png", cv::IMREAD_GRAYSCALE);

	const size_t MapWidth = inputImg.cols;
	const size_t MapHeight = inputImg.rows;
	const float SedimentVisualizeFactor = 10.0f;

	cv::Mat heightmap(MapHeight, MapWidth, CV_32F, cv::Scalar(1.0f));
	cv::Mat water_map(MapHeight, MapWidth, CV_32F, cv::Scalar(0.0f));
	cv::Mat sediment_map(MapHeight, MapWidth, CV_32F, cv::Scalar(0.0f));

	for (int y = 0; y < MapHeight; y++) 
	{
		for (int x = 0; x < MapWidth; x++) 
		{
			const uint8_t pixel = inputImg.at<uint8_t>(y, x);

			// Add a normalized water value (higher for lower values and lower for higher values)
			if (pixel < WaterThresholdValue)
				water_map.at<float>(y, x) = 1.0f - (pixel / (float)WaterThresholdValue);
		}
	}

	if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;

	const size_t bufferDataSize = MapWidth * MapHeight * sizeof(float);
	cl_mem heightMapBuffer = OpenCLUtils::create_input_buffer(context, heightmap.data, bufferDataSize);
	cl_mem waterMapBuffer = OpenCLUtils::create_input_buffer(context, water_map.data, bufferDataSize);
	cl_mem sedimentMapBuffer = OpenCLUtils::create_input_buffer(context, sediment_map.data, bufferDataSize);

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &heightMapBuffer);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &waterMapBuffer);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &sedimentMapBuffer);
	err |= clSetKernelArg(kernel, 3, sizeof(float), &ErosionRate);
	err |= clSetKernelArg(kernel, 4, sizeof(float), &DepositionRate);
	err |= clSetKernelArg(kernel, 5, sizeof(float), &FlowSpeed);
	err |= clSetKernelArg(kernel, 6, sizeof(int), &MapWidth);
	err |= clSetKernelArg(kernel, 7, sizeof(int), &MapHeight);
	if (err < 0)
	{
		perror("Couldn't create a kernel argument");
		return false;
	}

	size_t global[2] = { MapWidth, MapHeight };

	const std::string winName = "Erosion Simulation";
	cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

	cv::Mat outputImg(MapHeight, MapWidth * 3, CV_32F, cv::Scalar(0));
	cv::imshow(winName, outputImg);

	Timer gpuBufferReadTimer;
	Timer drawTimer;

	while (true)
	{
		gpuBufferReadTimer.Start();

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
								  heightMapBuffer,
								  CL_FALSE,
								  0,
								  bufferDataSize,
								  heightmap.data,
								  0,
								  NULL,
								  NULL);

		err |= clEnqueueReadBuffer(queue,
								   sedimentMapBuffer,
								   CL_FALSE,
								   0,
								   bufferDataSize,
								   sediment_map.data,
								   0,
								   NULL,
								   NULL);

		if (err < 0)
		{
			perror("Couldn't read the buffer");
			return false;
		}

		clFinish(queue);

		const double gpuBufferTime_ms = gpuBufferReadTimer.Elapsed_ms();

		// Visualization logic
		drawTimer.Start();

		// Apply sediment visualization factor
		sediment_map *= SedimentVisualizeFactor;

		heightmap.copyTo(outputImg(cv::Rect(0, 0, heightmap.cols, heightmap.rows)));
		sediment_map.copyTo(outputImg(cv::Rect(MapWidth, 0, sediment_map.cols, sediment_map.rows)));
		water_map.copyTo(outputImg(cv::Rect(MapWidth * 2, 0, water_map.cols, water_map.rows)));

		cv::imshow(winName, outputImg);

		// Press 'ESC' to exit
		if (cv::waitKey(1) == 27)
		{
			break;
		}

		const double drawTime_ms = drawTimer.Elapsed_ms();

		std::cout << "GPU Read Time: " << std::to_string(gpuBufferTime_ms) << "\tDraw Time: " << std::to_string(drawTime_ms) << std::endl;
	}

	cv::destroyWindow(winName);

    ///* Deallocate resources */
    
	clReleaseKernel(kernel);
	clReleaseMemObject(heightMapBuffer);
	clReleaseMemObject(waterMapBuffer);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}