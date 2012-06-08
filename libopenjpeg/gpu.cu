#include<stdio.h>
#include<malloc.h>
#include<cuda.h>

#include "opj_includes.h"

__global__ void kernel_dc_level_shift(int *d_current_ptr, int m_dc_level_shift, int min, int max, int stride, int height, int width, int qmfbid, int size) {

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;

	if(threadID > size) return; 

	int offset = threadID % (width+stride);

	if(offset < width) {
		int value = d_current_ptr[threadID];
		if(qmfbid != 1) {
			float f = *((float *)&value);
			value = lrintf(f);
		}
		value+=m_dc_level_shift;
		if(value < min) {
			d_current_ptr[threadID] = min;
		} else if (value > max) {
			d_current_ptr[threadID] = max;
		} else { 
			d_current_ptr[threadID] = value;
		}
	}
}



void gpu_dc_level_shift_decode( OPJ_INT32 *current_ptr,  OPJ_INT32 m_dc_level_shift,  OPJ_INT32 min, OPJ_INT32 max, OPJ_UINT32 stride, 
	OPJ_UINT32 height, OPJ_UINT32 width, OPJ_INT32 qmfbid ) { 

	OPJ_INT32 size = height*(width+stride);

	/* Debug Statement
	printf("Min %d, Max %d, Height %d, Width %d, Stride %d, qmfbid %d, shift %d\n",
			min,max,height,width,stride,qmfbid,m_dc_level_shift);
	*/

	OPJ_INT32 numBlocks = 1;
	OPJ_INT32 numThreadsPerBlock = size/numBlocks;

	if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
		numBlocks = (OPJ_INT32)ceil(size/(OPJ_FLOAT32)MAX_THREADS_PER_BLOCK);
		numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
	}

	dim3 grid(numBlocks, 1, 1);
	dim3 threads(numThreadsPerBlock, 1, 1); 

	OPJ_INT32 *d_current_ptr;
	cudaMalloc((OPJ_INT32 **)&d_current_ptr, sizeof(OPJ_INT32)*size);
	cudaMemcpy(d_current_ptr, current_ptr, sizeof(OPJ_INT32)*size, cudaMemcpyHostToDevice);
	
	cudaThreadSynchronize();

	kernel_dc_level_shift<<<grid, threads, 0>>>(d_current_ptr, m_dc_level_shift, min, max, stride, height, width, qmfbid, size);

	cudaThreadSynchronize();

	cudaMemcpy(current_ptr, d_current_ptr, sizeof(OPJ_INT32)*size, cudaMemcpyDeviceToHost);

}
