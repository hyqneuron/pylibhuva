void CPU_spatial_gather_hwc(float* compact, float* sparse, int*indices, int C, int H, int W, int K);
void CPU_spatial_scatter_hwc(float* compact, float* sparse, int*indices, int C, int H, int W, int K);

void launch_spatial_gather_hwc(float* compact, float*sparse, int* indices, int C, int H, int W, int K);
void launch_spatial_scatter_hwc(float* compact, float*sparse, int* indices, int C, int H, int W, int K);

void CPU_depthwise_conv2d_chw_k3(float *input, float* weight, float* output, int N, int C, int H, int W);
void launch_depthwise_conv2d_chw_k3(float* input, float* weight, float*output, int N, int C, int H, int W, bool perregion);
