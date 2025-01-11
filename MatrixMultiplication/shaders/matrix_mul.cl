__kernel void matrix_mul(__global const float* a,
                         __global const float* b,
                         int M, // Rows in A
                         int N, // Columns in A (Rows in B)
                         int K, // Columns in B
                         __global float* result) {

    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < K) 
    {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) 
        {
            sum += a[row * N + k] * b[k * K + col];
        }
        result[row * K + col] = sum;
    }
}
