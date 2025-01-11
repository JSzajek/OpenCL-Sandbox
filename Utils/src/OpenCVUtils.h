#pragma once

#include <opencv2/opencv.hpp>

class OpenCVUtils
{
public:
	static bool ConvertType(cv::Mat& input, 
							int targetType)
	{
		if (input.type() != targetType)
		{
			switch (targetType)
			{
				case CV_8UC1:
				{
					if (input.channels() == 3)
					{
						cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
						cv::cvtColor(input, input, cv::COLOR_RGB2GRAY);
						return true;
					}
					else if (input.channels() == 4)
					{
						cv::cvtColor(input, input, cv::COLOR_RGBA2BGR);
						cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
						cv::cvtColor(input, input, cv::COLOR_RGB2GRAY);
						return true;
					}
					break;
				}
				case CV_8UC3:
				{
					if (input.channels() == 1)
					{
						cv::cvtColor(input, input, cv::COLOR_GRAY2BGR);
						return true;
					}
					else if (input.channels() == 4)
					{
						cv::cvtColor(input, input, cv::COLOR_RGBA2BGR);
						return true;
					}
					break;
				}
				case CV_8UC4:
				{
					if (input.channels() == 1)
					{
						cv::cvtColor(input, input, cv::COLOR_GRAY2BGRA);
						return true;
					}
					if (input.channels() == 3)
					{
						cv::cvtColor(input, input, cv::COLOR_BGR2BGRA);
						return true;
					}
					break;
				}
				default:
					return false;
			}
		}
		return true;
	}
};