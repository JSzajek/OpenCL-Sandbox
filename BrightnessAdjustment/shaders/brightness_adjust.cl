__kernel void adjust(__global const uchar4* input,
                     float factor,
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
        output[index] = (uchar4)(clamp(pixel.x * factor, 0.0f, 255.0f),
                                 clamp(pixel.y * factor, 0.0f, 255.0f),
                                 clamp(pixel.z * factor, 0.0f, 255.0f),
                                 pixel.w);
    }
}
