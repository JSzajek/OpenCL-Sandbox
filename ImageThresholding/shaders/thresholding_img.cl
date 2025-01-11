__kernel void threshold(__global const uchar4* input,
                        uchar threshold,
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

        // Convert to grayscale using luminance formula
        uchar gray = (uchar)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);

        // Apply thresholding
        uchar new_color = (gray > threshold) ? 255 : 0;
        output[index] = (uchar4)(new_color, new_color, new_color, pixel.w);
    }
}
