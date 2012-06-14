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

__global__ void kernel_v4dwt_interleave_h(float *d_tilec_data, float4 *d_h_wavelet, int h_wavelet_sn, int h_wavelet_dn, int h_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rw) {

	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if(j < h_wavelet_sn) {
		d_h_wavelet[i*rw+(j*2)].x = d_tilec_data[i*4*w + j];
		d_h_wavelet[i*rw+(j*2)].y = d_tilec_data[(((4*i) + 1)*w) + j];
		d_h_wavelet[i*rw+(j*2)].z = d_tilec_data[(((4*i) + 2)*w) + j];
		d_h_wavelet[i*rw+(j*2)].w = d_tilec_data[(((4*i) + 3)*w) + j];    
 
	} else { 
		int k = j - h_wavelet_sn;
		d_h_wavelet[i*rw+(k*2 + 1)].x = d_tilec_data[i*4*w + j];
		d_h_wavelet[i*rw+(k*2 + 1)].y = d_tilec_data[(((4*i) + 1)*w) + j];
		d_h_wavelet[i*rw+(k*2 + 1)].z = d_tilec_data[(((4*i) + 2)*w) + j];
		d_h_wavelet[i*rw+(k*2 + 1)].w = d_tilec_data[(((4*i) + 3)*w) + j];     
	}

	/* Add boundary conditions */

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

static OPJ_UINT32 dwt_max_wavelet_size_v2(opj_tcd_resolution_v2_t* restrict r, OPJ_UINT32 i) {
	OPJ_UINT32 mr	= 0;
	OPJ_UINT32 w;
	while( --i ) {
		++r;
		if( mr < ( w = r->x1 - r->x0 ) )
			mr = w ;
		if( mr < ( w = r->y1 - r->y0 ) )
			mr = w ;
	}
	return mr ;
}



opj_bool gpu_dwt_decode_real_v2(opj_tcd_tilecomp_v2_t* restrict tilec, OPJ_UINT32 numres) { 

	opj_tcd_resolution_v2_t* res = tilec->resolutions;

	OPJ_UINT32 rw = res->x1 - res->x0;	/* width of the resolution level computed */
	OPJ_UINT32 rh = res->y1 - res->y0;	/* height of the resolution level computed */
	OPJ_UINT32 w = tilec->x1 - tilec->x0;
	
	OPJ_UINT32 bufsize = (tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0);
	OPJ_FLOAT32 *aj = (OPJ_FLOAT32 *) tilec->data;
	
	OPJ_FLOAT32 *d_tilec_data;
	cudaMalloc((OPJ_FLOAT32 **)&d_tilec_data, sizeof(OPJ_FLOAT32)*bufsize);
	cudaMemcpy(d_tilec_data, aj, sizeof(OPJ_FLOAT32)*bufsize, cudaMemcpyHostToDevice);
	
	int wavelet_size = dwt_max_wavelet_size_v2(res, numres) + 5;

	
	int h_wavelet_sn, h_wavelet_dn, h_wavelet_cas;
	int v_wavelet_sn, v_wavelet_dn, v_wavelet_cas;

	
	while( --numres) {

		h_wavelet_sn = rw;
		v_wavelet_sn = rh;

		++res;

		rw = res->x1 - res->x0;	/* width of the resolution level computed */
		rh = res->y1 - res->y0;	/* height of the resolution level computed */


		h_wavelet_dn = rw - h_wavelet_sn;
		h_wavelet_cas = res->x0 % 2;
		
		OPJ_UINT32 size_h = ceil(rh/4.0);
		OPJ_UINT32 size_w = rw;

		float4 *d_h_wavelet; 
		cudaMalloc((float4 **)&d_h_wavelet, sizeof(float4)*size_h*size_w);
		
		dim3 threads(MAX_THREADS_HEIGHT, MAX_THREADS_WIDTH, 1);
 
		dim3 grid(
			ceil(size_h/((float)(MAX_THREADS_HEIGHT))), 
			ceil(size_w/((float)(MAX_THREADS_WIDTH))),
			1
		);

		cudaThreadSynchronize();
		kernel_v4dwt_interleave_h<<<grid, threads, 0>>>(d_tilec_data, d_h_wavelet, h_wavelet_sn, h_wavelet_dn, h_wavelet_cas, w, bufsize, rw);
		cudaThreadSynchronize();

		/* FIXME Add and Fuse v4dwt_decode and v4dwt_interleave_h with above kernel */
		
	}

	return OPJ_TRUE;
}
