
__constant int Gx[3][3] = 
{
    { -1, 0, 1 },
    { -2, 0, 2 },
    { -1, 0, 1 }
};

__constant int Gy[3][3] = 
{
    { -1, -2, -1 },
    {  0,  0,  0 },
    {  1,  2,  1 }
};

__kernel void edge_detect(__global const uchar4* input,
                          int width,
                          int height,
                          __global uchar4* output) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) 
    {
        float gx = 0.0f;
        float gy = 0.0f;

        for (int fx = -1; fx <= 1; fx++) 
        {
            for (int fy = -1; fy <= 1; fy++) 
            {
                int nx = clamp(x + fx, 0, width - 1);
                int ny = clamp(y + fy, 0, height - 1);
                int index = ny * width + nx;
                uchar4 pixel = input[index];
                float intensity = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
                gx += intensity * Gx[fx + 1][fy + 1];
                gy += intensity * Gy[fx + 1][fy + 1];
            }
        }

        float magnitude = sqrt(gx * gx + gy * gy);
        int index = y * width + x;
        output[index] = (uchar)clamp(magnitude, 0.0f, 255.0f);
    }
}
