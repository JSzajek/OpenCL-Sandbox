__kernel void edge_detect(__global const uchar4* input,
                          float low_threshold,
                          float high_threshold,
                          int width,
                          int height,
                          __global uchar4* output) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) 
    {
        int index = y * width + x;
        uchar4 pixel = input[index];

        // Convert to grayscale intensity
        float intensity = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        // Apply Sobel operator (simplified for demonstration)
        int gx = 0;
        int gy = 0;
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) 
        {
            gx = -input[(y - 1) * width + (x - 1)].x - 2 * input[y * width + (x - 1)].x - input[(y + 1) * width + (x - 1)].x +
                  input[(y - 1) * width + (x + 1)].x + 2 * input[y * width + (x + 1)].x + input[(y + 1) * width + (x + 1)].x;

            gy = -input[(y - 1) * width + (x - 1)].x - 2 * input[(y - 1) * width + x].x - input[(y - 1) * width + (x + 1)].x +
                  input[(y + 1) * width + (x - 1)].x + 2 * input[(y + 1) * width + x].x + input[(y + 1) * width + (x + 1)].x;
        }

        // Calculate gradient magnitude
        float magnitude = sqrt((float)(gx * gx + gy * gy));

        // Apply non-maximum suppression and hysteresis (simplified)
        if (magnitude > high_threshold)
        {
            output[index] = (uchar4)(255, 255, 255, 255); // Strong edge
        }
        else if (magnitude < low_threshold)
        {
            output[index] = (uchar4)(0, 0, 0, 255); // Non-edge
        }
        else
        {
            output[index] = (uchar4)(128, 128, 128, 255); // Weak edge
        }
    }
}
