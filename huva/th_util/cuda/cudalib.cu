#include <math.h>
#include <assert.h>
#include <stdio.h> // debugging


extern "C" {

// ===================================================================================================================
// HWC-Gather-Scatter ================================================================================================
// ===================================================================================================================

void CPU_spatial_transfer_hwc(bool gather, float* compact, float* sparse, int* indices, int C, int H, int W, int K){
    /*
    Note the channel order we use is HWC. So tensor has to be permuted and made contiguous before being passed in

    compact: KxC
    sparse:  HxWxC
    indices: K
     */
    // k indexes location, c indexes channel
    for(int k = 0; k < K; k++){
        int s_index_outer = indices[k] * C; // index for sparse
        int c_index_outer = k * C;          // index for compact
        assert(indices[k] < H * W);
        for(int c = 0; c < C; c++){
            if(gather)                      // cpu can perfectly predict this branch, so never mind
                compact[c_index_outer + c] = sparse[s_index_outer + c];
            else
                sparse[s_index_outer + c] = compact[c_index_outer + c] ;
        }
    }
}

void CPU_spatial_gather_hwc(float* compact, float* sparse, int*indices, int C, int H, int W, int K){
    CPU_spatial_transfer_hwc(true, compact, sparse, indices, C, H, W, K);
}
void CPU_spatial_scatter_hwc(float* compact, float* sparse, int*indices, int C, int H, int W, int K){
    CPU_spatial_transfer_hwc(false, compact, sparse, indices, C, H, W, K);
}

__global__ void CUDA_spatial_gather_hwc(float* compact, float*sparse, int* indices, int C, int H, int W, int K){
    /*
    Note the channel order we use is HWC. So tensor has to be permuted and made contiguous before being passed in

    compact: KxC
    sparse:  HxWxC
    indices: K

    One block handles an entire location, assuming C <= 1024
        num_blocks = K
    One warp handles 32 channels
        num_threads = C

    Unimplemented:
        In the case of C > 1024, a location is handled using number of blocks = ceil(C / 1024)
            grid size = (K, ceil(C/1024))
     */
    int k = blockIdx.x;
    int c = threadIdx.x;
    int s_index = indices[k] * C + c;       // all threads in block access identical memory location
    int c_index = k * C + c;
    compact[c_index] = sparse[s_index];     // threads within a warp access contiguous memory regions 
}

__global__ void CUDA_spatial_scatter_hwc(float* compact, float*sparse, int* indices, int C, int H, int W, int K){
    int k = blockIdx.x;
    int c = threadIdx.x;
    int s_index = indices[k] * C + c;
    int c_index = k * C;
    sparse[s_index] = compact[c_index];
}

void launch_spatial_gather_hwc(float* compact, float*sparse, int* indices, int C, int H, int W, int K){
    assert(C <= 1024); // currently more than 1024 channels not supported
    CUDA_spatial_gather_hwc<<<K, C>>>(compact, sparse, indices, C, H, W, K);
}

void launch_spatial_scatter_hwc(float* compact, float*sparse, int* indices, int C, int H, int W, int K){
    assert(C <= 1024); // currently more than 1024 channels not supported
    CUDA_spatial_scatter_hwc<<<K, C>>>(compact, sparse, indices, C, H, W, K);
}

// ===================================================================================================================
// CHW-Gather-Scatter ================================================================================================
// ===================================================================================================================

// not implemented yet

// ===================================================================================================================
// Depthwise convolution on CHW format ===============================================================================
// ===================================================================================================================

void CPU_depthwise_conv2d_chw_k3(float *input, float* weight, float* output, int N, int C, int H, int W){
    /*
    Perform convolution on a per-channel basis (depthwise convolution) on NCHW input
    Currently only support k=3, stride=1, padding=1

    input : NxCxHxW
    output: NxCXHxW
    weight: Cx3x3

     */
    const int K = 3;    // kernel size
    const int off_y_init = -1;
    const int off_y_last = 1;
    const int off_x_init = -1;
    const int off_x_last = 1;
    for(int n = 0; n < N; n++){                 // for every n
        int offset0 = n * C * H * W;            // beginning of sample n
        for(int c = 0; c < C; c++){             // for every c
            int offset1 = offset0 + c * H * W;  // beginning of map c
            for(int h = 0; h < H; h++){         // for every h
                for(int w = 0; w < W; w++){     // for every w
                    float sum = 0.0;
                    // loop over kernel
                    for(int off_y = off_y_init; off_y <= off_y_last; off_y++){       // for every y offset
                        // zero pad:
                        int y = h + off_y;
                        if( y < 0 || y >= H)
                            continue;
                        int offset2 = offset1 + y * W;
                        for(int off_x = off_x_init; off_x <= off_x_last; off_x++){   // for every x offset
                            // zero pad:
                            int x = w + off_x;
                            if( x < 0 || x >= W)
                                continue;
                            int offset3 = offset2 + x;
                            // accumulate into sum
                            sum += weight[c*K*K+(off_y - off_y_init)*K + (off_x - off_x_init) ] * input[offset3];
                        }
                    }
                    output[offset1 + h * W + w] = sum;
                }
            }
        }
    }
}

void __global__ CUDA_depthwise_conv2d_chw_k3_perregion(
        float* input, float* weight, float*output, int N, int C, int H, int W, int grid_H, int grid_W){
    /*
    Perform convolution on a per-channel basis (depthwise convolution) on 1CHW input
    Currently only support k=3, stride=1, padding=1

    input : 1xCxHxW
    output: 1xCXHxW
    weight: Cx3x3

    One region per block
     */
    const int w = blockIdx.x * blockDim.x + threadIdx.x; 
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z / C;
    const int c = blockIdx.z % C;
    // kernel parameters
    const int K = 3;    // kernel size
    const int off_y_init = -1;
    const int off_y_last = 1;
    const int off_x_init = -1;
    const int off_x_last = 1;
    if( w >= W || h >= H )
        return;
    int offset0 = n * C * H * W + c * H * W;
    // loop over kernel
    float sum = 0.0;
    for(int off_y = off_y_init; off_y <= off_y_last; off_y++){
        int y = h + off_y;
        if( y < 0 || y >= H)
            continue;
        int offset1 = offset0 + y * W;
        for(int off_x = off_x_init; off_x <= off_x_last; off_x++){
            int x = w + off_x;
            if( x < 0 || x >= W)
                continue;
            int offset2 = offset1 + x;
            sum += weight[c*K*K+ (off_y - off_y_init)*K + (off_x - off_x_init)] * input[offset2];
        }
    }
    output[offset0 + h * W + w] = sum;
}

void __global__ CUDA_depthwise_conv2d_chw_k3_permap(
        float* input, float* weight, float*output, int N, int C, int H, int W, int grid_H, int grid_W){
    /*
    One feature map per block
     */
    const int c = blockIdx.x;
    const int n = blockIdx.y;
    const int h_off = threadIdx.y;
    const int w_off = threadIdx.x;
    // kernel parameters
    const int K = 3;
    const int off_y_init = -1;
    const int off_y_last = 1;
    const int off_x_init = -1;
    const int off_x_last = 1;
    int offset0 = n * C * H * W + c * H * W;
    // loop over grid
    for(int grid_h = 0; grid_h < grid_H; grid_h++){
        int h = grid_h * blockDim.y + h_off;
        if(h >= H)
            return;
        for(int grid_w = 0; grid_w < grid_W; grid_w++){
            int w = grid_w * blockDim.x + w_off;
            if (w >= W)
                break;
            // loop over kernel
            float sum = 0.0;
            for(int off_y = off_y_init; off_y <= off_y_last; off_y++){
                int y = h + off_y;
                if( y < 0 || y >= H)
                    continue;
                int offset1 = offset0 + y * W;
                for(int off_x = off_x_init; off_x <= off_x_last; off_x++){
                    int x = w + off_x;
                    if( x < 0 || x >= W)
                        continue;
                    int offset2 = offset1 + x;
                    sum += weight[c*K*K + (off_y - off_y_init)*K + (off_x - off_x_init)] * input[offset2];
                }
            }
            output[offset0 + h * W + w] = sum;
        }
    }
}

void launch_depthwise_conv2d_chw_k3(
        float* input, float* weight, float*output, int N, int C, int H, int W, bool perregion){
    /*
    Block partition:
    - partition feature map into 32x32 regions
    - one block handles one region of one feature map
    - if there are C maps and M regions, there'll be CxM blocks
      - M = grid_H * grid_W

    Alternative block partition:
    - partition feature map into 32x32 regions
    - one block handles one entire feature map
    - If there are C maps, there are C block
      - within-block striding to handle entire feature map
      - one block processes one region at a time, and loop through
     */
    int grid_H = (H+31) / 32;   // 32 vertical pixels per region
    int grid_W = (W+31) / 32;   // 32 horizontal pixels per region
    dim3 grid_size(grid_W, grid_H, C*N);
    dim3 block_size(32, 32);            // each block covers 32x32 region
    if(perregion){
        CUDA_depthwise_conv2d_chw_k3_perregion<<<grid_size, block_size>>>(
                input, weight, output, N, C, H, W, grid_H, grid_W);
    }
    else{
        grid_size = dim3(C, N);
        CUDA_depthwise_conv2d_chw_k3_permap   <<<grid_size, block_size>>>(
                input, weight, output, N, C, H, W, grid_H, grid_W);
    }
}

} // extern "C"
