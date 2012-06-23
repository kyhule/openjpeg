#include<stdio.h>
#include<cuda.h>
#include "opj_includes.h"

static const float dwt_alpha =  1.586134342f; /*  12994 */
static const float dwt_beta  =  0.052980118f; /*    434 */
static const float dwt_gamma = -0.882911075f; /*  -7233 */
static const float dwt_delta = -0.443506852f; /*  -3633 */

static const float K      = 1.230174105f; /*  10078 */
static const float c13318 = 1.625732422f;

__device__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ float4 operator*(const float4 &a, const float4 &b) {
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}


__device__ INLINE int device_int_min(int a, int b) {
        return a < b ? a : b;
}

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
__global__ void kernel_v4dwt_h_cas1(float *d_tilec_data, int h_wavelet_sn, int h_wavelet_dn, int h_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rw, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta) {

	unsigned int i = blockIdx.x;
	unsigned int j = threadIdx.x; 

	__shared__ float4 shared_h_wavelet[512];
		
	__syncthreads();  

	if(j < h_wavelet_sn) {
		shared_h_wavelet[((2*j)+1)].x = const1 * d_tilec_data[i*4*w + j];
		shared_h_wavelet[((2*j)+1)].y = const1 * d_tilec_data[(((4*i) + 1)*w) + j];
		shared_h_wavelet[((2*j)+1)].z = const1 * d_tilec_data[(((4*i) + 2)*w) + j];
		shared_h_wavelet[((2*j)+1)].w = const1 * d_tilec_data[(((4*i) + 3)*w) + j];    
	} else { 
		int p = j - h_wavelet_sn;
		shared_h_wavelet[p*2].x = const2 * d_tilec_data[i*4*w + j];
		shared_h_wavelet[p*2].y = const2 * d_tilec_data[(((4*i) + 1)*w) + j];
		shared_h_wavelet[p*2].z = const2 * d_tilec_data[(((4*i) + 2)*w) + j];
		shared_h_wavelet[p*2].w = const2 * d_tilec_data[(((4*i) + 3)*w) + j];     
	}
	__syncthreads();

	int a = 1;
	int b = 0;

	int k, m;

	// start at 0/2 case 
	k = h_wavelet_sn; 
	m = h_wavelet_dn - a;
	if(j%2==1) {
		if(j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_delta);
		} else if(j < 2*k) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m]*(f4_dwt_delta + f4_dwt_delta));
		}
	}

	__syncthreads();
	

	// start at 1/1 case 
	k = h_wavelet_dn;
	m = h_wavelet_sn - b;
	if(j%2 == 0) { 
		if(j > 0 && j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_gamma);
		} else if(j == 0){ 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j+1] + shared_h_wavelet[j+1])*f4_dwt_gamma);
		} else if(j < 2*k){
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m-1]*(f4_dwt_gamma + f4_dwt_gamma)); 
		}

	}
	__syncthreads();

	// start at 0/2 case
	k = h_wavelet_sn; 
	m = h_wavelet_dn - a;
	if(j%2==1) {
		if(j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_beta);
		} else if(j < 2*k) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m]*(f4_dwt_beta + f4_dwt_beta));
		}

	}
	__syncthreads();

	// start at 1/1 case 
	k = h_wavelet_dn;
	m = h_wavelet_sn - b;
	if(j%2 == 0) { 
		if(j > 0 && j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_alpha);
		} else if(j==0){ 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j+1] + shared_h_wavelet[j+1])*f4_dwt_alpha);
		} else if(j < 2*k){
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m-1]*(f4_dwt_alpha + f4_dwt_alpha)); 
		}
	}
	__syncthreads(); 
	
	d_tilec_data[i*4*w + j] = shared_h_wavelet[j].x;
	d_tilec_data[(((4*i) + 1)*w) + j] = shared_h_wavelet[j].y;
	d_tilec_data[(((4*i) + 2)*w) + j] = shared_h_wavelet[j].z;
	d_tilec_data[(((4*i) + 3)*w) + j] = shared_h_wavelet[j].w;
}

__global__ void kernel_v4dwt_h_cas0(float *d_tilec_data, int h_wavelet_sn, int h_wavelet_dn, int h_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rw, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta) {

	unsigned int i = blockIdx.x;
	unsigned int j = threadIdx.x; 

	__shared__ float4 shared_h_wavelet[512];
		
	__syncthreads();  

	if(j < h_wavelet_sn) {
		shared_h_wavelet[2*j].x = const1 * d_tilec_data[i*4*w + j];
		shared_h_wavelet[2*j].y = const1 * d_tilec_data[(((4*i) + 1)*w) + j];
		shared_h_wavelet[2*j].z = const1 * d_tilec_data[(((4*i) + 2)*w) + j];
		shared_h_wavelet[2*j].w = const1 * d_tilec_data[(((4*i) + 3)*w) + j];    
	} else { 
		int p = j - h_wavelet_sn;
		shared_h_wavelet[(p*2 + 1)].x = const2 * d_tilec_data[i*4*w + j];
		shared_h_wavelet[(p*2 + 1)].y = const2 * d_tilec_data[(((4*i) + 1)*w) + j];
		shared_h_wavelet[(p*2 + 1)].z = const2 * d_tilec_data[(((4*i) + 2)*w) + j];
		shared_h_wavelet[(p*2 + 1)].w = const2 * d_tilec_data[(((4*i) + 3)*w) + j];     
	}
	__syncthreads();

	int a = 0;
	int b = 1;

	int k, m;

	/* start at 1/1 case */
	k = h_wavelet_sn;
	m = h_wavelet_dn - a;
	if(j%2 == 0) { 
		if(j > 0 && j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_delta);
		} else if(j == 0){ 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j+1] + shared_h_wavelet[j+1])*f4_dwt_delta);
		} else if(j < 2*k){
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m-1]*(f4_dwt_delta + f4_dwt_delta)); 
		}

	}
	__syncthreads();

	/* start at 0/2 case */
	k = h_wavelet_dn; 
	m = h_wavelet_sn - b;
	if(j%2==1) {
		if(j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_gamma);
		} else if(j < 2*k) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m]*(f4_dwt_gamma + f4_dwt_gamma));
		}
	}

	__syncthreads();

	/* start at 1/1 case */
	k = h_wavelet_sn;
	m = h_wavelet_dn - a;
	if(j%2 == 0) { 
		if(j > 0 && j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_beta);
		} else if(j==0){ 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j+1] + shared_h_wavelet[j+1])*f4_dwt_beta);
		} else if(j < 2*k){
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m-1]*(f4_dwt_beta + f4_dwt_beta)); 
		}
	}
	__syncthreads();

	/* start at 0/2 case */
	k = h_wavelet_dn; 
	m = h_wavelet_sn - b;
	if(j%2==1) {
		if(j < 2*m) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + ((shared_h_wavelet[j-1] + shared_h_wavelet[j+1])*f4_dwt_alpha);
		} else if(j < 2*k) { 
			shared_h_wavelet[j] = shared_h_wavelet[j] + (shared_h_wavelet[2*m]*(f4_dwt_alpha + f4_dwt_alpha));
		}

	}
	__syncthreads();
	
	d_tilec_data[i*4*w + j] = shared_h_wavelet[j].x;
	d_tilec_data[(((4*i) + 1)*w) + j] = shared_h_wavelet[j].y;
	d_tilec_data[(((4*i) + 2)*w) + j] = shared_h_wavelet[j].z;
	d_tilec_data[(((4*i) + 3)*w) + j] = shared_h_wavelet[j].w;
}

__global__ void kernel_v4dwt_v_cas1(float *d_tilec_data, int v_wavelet_sn, int v_wavelet_dn, int v_wavelet_cas, unsigned int w, unsigned int buffsize, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta) {

	unsigned int i = threadIdx.x; // 0 to rh
	unsigned int j = blockIdx.x; // 0 to rw/4

	__shared__ float4 shared_v_wavelet[512];
		
	__syncthreads(); 
 
	if(i < v_wavelet_sn) {
		shared_v_wavelet[((2*i)+1)].x = const1 * d_tilec_data[i*w + (4*j)];
		shared_v_wavelet[((2*i)+1)].y = const1 * d_tilec_data[i*w + (4*j) + 1];
		shared_v_wavelet[((2*i)+1)].z = const1 * d_tilec_data[i*w + (4*j) + 2];
		shared_v_wavelet[((2*i)+1)].w = const1 * d_tilec_data[i*w + (4*j) + 3];    
	} else { 
		int p = i - v_wavelet_sn;
		shared_v_wavelet[p*2].x = const2 * d_tilec_data[i*w + (4*j)];
		shared_v_wavelet[p*2].y = const2 * d_tilec_data[i*w + (4*j) + 1];
		shared_v_wavelet[p*2].z = const2 * d_tilec_data[i*w + (4*j) + 2];
		shared_v_wavelet[p*2].w = const2 * d_tilec_data[i*w + (4*j) + 3];     
	}
	
	__syncthreads();

	int a = 1;
	int b = 0;

	int k, m;

	// start at 0/2 case
	k = v_wavelet_sn;
	m = v_wavelet_dn - a;
	if(i%2==1) {
		if(i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_delta);
		} else if(i < 2*k) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m]*(f4_dwt_delta + f4_dwt_delta));
		}
	}

	__syncthreads();

	
	// start at 1/1 case 
	k = v_wavelet_dn;
	m = v_wavelet_sn - b;
	if(i%2 == 0) { 
		if(i > 0 && i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_gamma);
		} else if(i == 0){ 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i+1] + shared_v_wavelet[i+1])*f4_dwt_gamma);
		} else if(i < 2*k){
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m-1]*(f4_dwt_gamma + f4_dwt_gamma)); 
		}

	}
	__syncthreads();


	// start at 0/2 case 
	k = v_wavelet_sn; 
	m = v_wavelet_dn - a;
	if(i%2==1) {
		if(i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_beta);
		} else if(i < 2*k) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m]*(f4_dwt_beta + f4_dwt_beta));
		}
	}
	__syncthreads();   

	// start at 1/1 case
	k = v_wavelet_dn;
	m = v_wavelet_sn - b;
	if(i%2 == 0) { 
		if(i > 0 && i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_alpha);
		} else if(i == 0){ 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i+1] + shared_v_wavelet[i+1])*f4_dwt_alpha);
		} else if(i < 2*k){
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m-1]*(f4_dwt_alpha + f4_dwt_alpha)); 
		}

	}
	__syncthreads();

	d_tilec_data[i*w + (4*j)] = shared_v_wavelet[i].x;
	d_tilec_data[i*w + (4*j) + 1] = shared_v_wavelet[i].y;
	d_tilec_data[i*w + (4*j) + 2] = shared_v_wavelet[i].z;
	d_tilec_data[i*w + (4*j) + 3] = shared_v_wavelet[i].w; 
}

__global__ void kernel_v4dwt_v_cas0(float *d_tilec_data, int v_wavelet_sn, int v_wavelet_dn, int v_wavelet_cas, unsigned int w, unsigned int buffsize, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta) {

	unsigned int i = threadIdx.x; // 0 to rh
	unsigned int j = blockIdx.x; // 0 to rw/4

	__shared__ float4 shared_v_wavelet[512];
		
	__syncthreads(); 
 
	if(i < v_wavelet_sn) {
		shared_v_wavelet[2*i].x = const1 * d_tilec_data[i*w + (4*j)];
		shared_v_wavelet[2*i].y = const1 * d_tilec_data[i*w + (4*j) + 1];
		shared_v_wavelet[2*i].z = const1 * d_tilec_data[i*w + (4*j) + 2];
		shared_v_wavelet[2*i].w = const1 * d_tilec_data[i*w + (4*j) + 3];    
	} else { 
		int p = i - v_wavelet_sn;
		shared_v_wavelet[(p*2 + 1)].x = const2 * d_tilec_data[i*w + (4*j)];
		shared_v_wavelet[(p*2 + 1)].y = const2 * d_tilec_data[i*w + (4*j) + 1];
		shared_v_wavelet[(p*2 + 1)].z = const2 * d_tilec_data[i*w + (4*j) + 2];
		shared_v_wavelet[(p*2 + 1)].w = const2 * d_tilec_data[i*w + (4*j) + 3];     
	}
	
	__syncthreads();

	int a = 0;
	int b = 1;

	int k, m;


	/* start at 1/1 case */
	k = v_wavelet_sn;
	m = v_wavelet_dn - a;
	if(i%2 == 0) { 
		if(i > 0 && i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_delta);
		} else if(i == 0){ 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i+1] + shared_v_wavelet[i+1])*f4_dwt_delta);
		} else if(i < 2*k){
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m-1]*(f4_dwt_delta + f4_dwt_delta)); 
		}

	}
	__syncthreads();

	/* start at 0/2 case */
	k = v_wavelet_dn; 
	m = v_wavelet_sn - b;
	if(i%2==1) {
		if(i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_gamma);
		} else if(i < 2*k) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m]*(f4_dwt_gamma + f4_dwt_gamma));
		}
	}
	__syncthreads();

	/* start at 1/1 case */
	k = v_wavelet_sn;
	m = v_wavelet_dn - a;
	if(i%2 == 0) { 
		if(i > 0 && i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_beta);
		} else if(i == 0){ 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i+1] + shared_v_wavelet[i+1])*f4_dwt_beta);
		} else if(i < 2*k){
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m-1]*(f4_dwt_beta + f4_dwt_beta)); 
		}

	}
	__syncthreads();

	/* start at 0/2 case */
	k = v_wavelet_dn; 
	m = v_wavelet_sn - b;
	if(i%2==1) {
		if(i < 2*m) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + ((shared_v_wavelet[i-1] + shared_v_wavelet[i+1])*f4_dwt_alpha);
		} else if(i < 2*k) { 
			shared_v_wavelet[i] = shared_v_wavelet[i] + (shared_v_wavelet[2*m]*(f4_dwt_alpha + f4_dwt_alpha));
		}
	}
	__syncthreads();   

	d_tilec_data[i*w + (4*j)] = shared_v_wavelet[i].x;
	d_tilec_data[i*w + (4*j) + 1] = shared_v_wavelet[i].y;
	d_tilec_data[i*w + (4*j) + 2] = shared_v_wavelet[i].z;
	d_tilec_data[i*w + (4*j) + 3] = shared_v_wavelet[i].w; 
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

	/* DEBUG  
	float *result = (float *)opj_malloc(sizeof(float)*bufsize); */
	
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

		/* FIXME add for loop to the 4-decode kernels, for dimension greater than 512
		if(rw > 512 || rh > 512) { 
			break;
		} */

		h_wavelet_dn = rw - h_wavelet_sn;
		h_wavelet_cas = res->x0 % 2;
		
		float4 f4_dwt_alpha = make_float4(dwt_alpha, dwt_alpha, dwt_alpha, dwt_alpha);
		float4 f4_dwt_beta = make_float4(dwt_beta, dwt_beta, dwt_beta, dwt_beta);
		float4 f4_dwt_gamma = make_float4(dwt_gamma, dwt_gamma, dwt_gamma, dwt_gamma);
		float4 f4_dwt_delta = make_float4(dwt_delta, dwt_delta, dwt_delta, dwt_delta);
		
		dim3 threads_h(rw, 1, 1);
		dim3 grid_h(ceil(rh/4.0), 1, 1);

		cudaThreadSynchronize();
		if(h_wavelet_cas == 0) { 
			kernel_v4dwt_h_cas0<<<grid_h, threads_h, 0>>>(d_tilec_data, h_wavelet_sn, h_wavelet_dn, h_wavelet_cas, w, bufsize, rw, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);
		
		} else { 
			kernel_v4dwt_h_cas1<<<grid_h, threads_h, 0>>>(d_tilec_data, h_wavelet_sn, h_wavelet_dn, h_wavelet_cas, w, bufsize, rw, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);

		}
		cudaThreadSynchronize();


		/* DEBUG v4dwt_h
		OPJ_INT32 j,k;
		cudaMemcpy(result, d_tilec_data, sizeof(OPJ_FLOAT32)*bufsize, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		for(j = rh; j > 3; j -= 4) {
			for(k = rw; --k >= 0;){
				printf("(%d,%d,%d,%d)\n",(int)(floor(result[k])),(int)(floor(result[k+w])),(int)(floor(result[k+w*2])),(int)(floor(result[k+w*3])));
			}
			result+=w*4;
		} */
		
		v_wavelet_dn = rh - v_wavelet_sn;
		v_wavelet_cas = res->y0 % 2;

		dim3 threads_v(rh, 1, 1);
		dim3 grid_v(ceil(rw/4.0),1,1);


		cudaThreadSynchronize();
		if(v_wavelet_cas == 0) { 
			kernel_v4dwt_v_cas0<<<grid_v, threads_v, 0>>>(d_tilec_data, v_wavelet_sn, v_wavelet_dn, v_wavelet_cas, w, bufsize, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);
		} else { 
			kernel_v4dwt_v_cas0<<<grid_v, threads_v, 0>>>(d_tilec_data, v_wavelet_sn, v_wavelet_dn, v_wavelet_cas, w, bufsize, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);
		}
		cudaThreadSynchronize();

		/* DEBUG v4dwt_v
		cudaMemcpy(result, d_tilec_data, sizeof(OPJ_FLOAT32)*bufsize, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		//OPJ_INT32 j,k;
		for(j = rw; j > 3; j-=4) { 
			for(k = 0; k < rh; ++k){
				printf("{%d,%d,%d,%d}\n",(int)(floor(result[k*w])),(int)(floor(result[k*w + 1])),(int)(floor(result[k*w + 2])),(int)(floor(result[k*w + 3])));
			}
			result+=4;
		} */ 

	}
	cudaFree(d_tilec_data);

	return OPJ_TRUE;
}
