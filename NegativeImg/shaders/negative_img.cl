__kernel void negative(__global const uchar4* input,
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

        output[index] = (uchar4)(255 - pixel.x, 
                                 255 - pixel.y, 
                                 255 - pixel.z, 
                                 pixel.w);
    }
}
