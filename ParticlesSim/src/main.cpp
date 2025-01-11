#include "OpenCLUtils.h"
#include "OpenCVUtils.h"
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

struct Vector4f
{
public:
	float x = 0;
	float y = 0;
	float z = 0;
	float w = 0;
};

struct Vector2f
{
public:
	Vector2f() = default;

	Vector2f(float _x, float _y)
		: x(_x), 
		y(_y)
	{
	}
public:
	float x = 0;
	float y = 0;
};

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
	program = OpenCLUtils::build_program(context, device, "shaders/particles.cl");
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
	constexpr size_t Num_Particles = 1000;

	const size_t ParticleRadius = 2;
	const float StartVelocityMagnitude = 1;

	float DeltaTime = 0.033f;
	Vector2f Gravity(0, 9.8f);
	float BounceFactor = 0.9f;

	const size_t MaxX = 512;
	const float HalfMaxX = MaxX * 0.5f;
	const size_t MaxY = 512;
	const float HalfMaxY = MaxY * 0.5f;

	Vector4f Bounds;
	Bounds.x = 0;			// MinX
	Bounds.y = MaxX;		// MaxX
	Bounds.z = 0;			// MinY
	Bounds.w = MaxY;		// MaxY

	std::vector<Vector2f> Positions(Num_Particles);
	std::vector<Vector2f> Velocities(Num_Particles);

	// Initialize particles
	for (int i = 0; i < Num_Particles; i++)
	{
		Positions[i]	= Vector2f(rand() % MaxX, 
								   rand() % MaxY);
		Velocities[i]	= Vector2f((rand() % MaxX - HalfMaxX) / HalfMaxX * StartVelocityMagnitude,
								   (rand() % MaxY - HalfMaxY) / HalfMaxY * StartVelocityMagnitude);
	}

	if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;


	const size_t bufferDataSize = Num_Particles * sizeof(Vector2f);

	cl_mem positionsBuffer = OpenCLUtils::create_input_buffer(context, Positions.data(), bufferDataSize);
	cl_mem velocitiesBuffer = OpenCLUtils::create_input_buffer(context, Velocities.data(), bufferDataSize);

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &positionsBuffer);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &velocitiesBuffer);
	err |= clSetKernelArg(kernel, 2, sizeof(float), &DeltaTime);
	err |= clSetKernelArg(kernel, 3, sizeof(Vector2f), &Gravity);
	err |= clSetKernelArg(kernel, 4, sizeof(Vector4f), &Bounds);
	err |= clSetKernelArg(kernel, 5, sizeof(float), &BounceFactor);
	if (err < 0)
	{
		perror("Couldn't create a kernel argument");
		return false;
	}

	size_t global = Num_Particles;

	const std::string winName = "Particles Simulation";
	cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
	cv::Mat outputImg(MaxY, MaxX, CV_8UC4, cv::Scalar(0, 0, 0));

	Timer gpuBufferReadTimer;
	Timer drawTimer;
	auto previousTime = std::chrono::high_resolution_clock::now();

	float deltaTime_s = 0;
	while (true)
	{
		gpuBufferReadTimer.Start();

		// Update delta time --------------------------------------------------
		err = clSetKernelArg(kernel, 2, sizeof(float), &deltaTime_s);
		if (err < 0)
		{
			perror("Couldn't create a kernel argument");
			return false;
		}
		// --------------------------------------------------------------------

		err = clEnqueueNDRangeKernel(queue,
									 kernel,
									 1,
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
								  positionsBuffer,
								  CL_TRUE, // Blocking read
								  0,
								  bufferDataSize,
								  Positions.data(),
								  0,
								  NULL,
								  NULL);
		if (err < 0)
		{
			perror("Couldn't read the buffer");
			return false;
		}
		const double gpuBufferTime_ms = gpuBufferReadTimer.Elapsed_ms();

		// Visualization logic
		drawTimer.Start();
		{
			outputImg.setTo(cv::Scalar(0, 0, 0));
			for (size_t x = 0; x < Num_Particles; ++x)
			{
				cv::circle(outputImg, cv::Point(Positions[x].x, Positions[x].y), ParticleRadius, cv::Scalar(255, 0, 0), 1, -1);
			}

			cv::imshow(winName, outputImg);

			// Press 'ESC' to exit
			if (cv::waitKey(1) == 27)
			{
				break;
			}
		}

		const double drawTime_ms = drawTimer.Elapsed_ms();

		std::cout << "GPU Read Time: " << std::to_string(gpuBufferTime_ms) << "\tDraw Time: " << std::to_string(drawTime_ms) << std::endl;
		deltaTime_s = (gpuBufferTime_ms + drawTime_ms) * 0.01f; // Convert back to seconds
	}

	cv::destroyWindow(winName);

    ///* Deallocate resources */
    
	clReleaseKernel(kernel);
	clReleaseMemObject(positionsBuffer);
	clReleaseMemObject(velocitiesBuffer);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}