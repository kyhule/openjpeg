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


opj_bool gpu_dc_level_shift_decode(opj_tcd_v2_t *p_tcd) { 

	OPJ_UINT32 compno;
	opj_tcd_tilecomp_v2_t * l_tile_comp = 00;
	opj_tccp_t * l_tccp = 00;
	opj_image_comp_t * l_img_comp = 00;
	opj_tcd_resolution_v2_t* l_res = 00;
	opj_tcd_tile_v2_t * l_tile;
	OPJ_UINT32 l_width,l_height;
	OPJ_INT32 * l_current_ptr;
	OPJ_INT32 l_min, l_max;
	OPJ_UINT32 l_stride;

	l_tile = p_tcd->tcd_image->tiles;
	l_tile_comp = l_tile->comps;
	l_tccp = p_tcd->tcp->tccps;
	l_img_comp = p_tcd->image->comps;

	for (compno = 0; compno < l_tile->numcomps; compno++) {

		l_res = l_tile_comp->resolutions + l_img_comp->resno_decoded;
		l_width = (l_res->x1 - l_res->x0);
		l_height = (l_res->y1 - l_res->y0);
		l_stride = (l_tile_comp->x1 - l_tile_comp->x0) - l_width;

		l_current_ptr = l_tile_comp->data;

		OPJ_INT32 size = l_height*(l_width+l_stride);

		OPJ_INT32 *d_current_ptr;
		cudaMalloc((OPJ_INT32 **)&d_current_ptr, sizeof(OPJ_INT32)*size);
		cudaMemcpy(d_current_ptr, l_current_ptr, sizeof(OPJ_INT32)*size, cudaMemcpyHostToDevice);


		OPJ_INT32 numBlocks = 1;
		OPJ_INT32 numThreadsPerBlock = size/numBlocks;
		if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
			numBlocks = (OPJ_INT32)ceil(size/(OPJ_FLOAT32)MAX_THREADS_PER_BLOCK);
			numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
		}

		dim3 grid(numBlocks, 1, 1);
		dim3 threads(numThreadsPerBlock, 1, 1); 

		if (l_img_comp->sgnd) {
			l_min = -(1 << (l_img_comp->prec - 1));
			l_max = (1 << (l_img_comp->prec - 1)) - 1;
		}
		else {
			l_min = 0;
			l_max = (1 << l_img_comp->prec) - 1;
		}


		cudaThreadSynchronize();

		kernel_dc_level_shift<<<grid, threads, 0>>>(d_current_ptr, l_tccp->m_dc_level_shift, l_min, l_max, l_stride, l_height, l_width, 
				l_tccp->qmfbid, size);

		cudaThreadSynchronize();

		cudaMemcpy(l_current_ptr, d_current_ptr, sizeof(OPJ_INT32)*size, cudaMemcpyDeviceToHost);

		++l_tile_comp;
		++l_img_comp;
		++l_tccp;
	}

	return OPJ_TRUE;
}

