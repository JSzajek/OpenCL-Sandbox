__kernel void rotate(__global const uchar4* input,
                     int width,
                     int height,
                     __global uchar4* output) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) 
    {
        int new_x = height - 1 - y;
        int new_y = x;
        int input_index = y * width + x;
        int output_index = new_y * height + new_x;
        output[output_index] = input[input_index];
    }
}
