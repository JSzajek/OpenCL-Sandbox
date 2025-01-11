__kernel void grayscale(__global const uchar* input,
                        int width,
                        int height,
                        __global uchar* output) 
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    int idx = (y * width + x) * 3;

    uchar r = input[idx];
    uchar g = input[idx + 1];
    uchar b = input[idx + 2];

    output[y * width + x] = (uchar)(0.3f * r + 0.59f * g + 0.11f * b);
}
