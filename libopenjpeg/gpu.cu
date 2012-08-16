#include<stdio.h>
#include<cuda.h>
#include "opj_includes.h"

static const float dwt_alpha =  1.586134342f; /*  12994 */
static const float dwt_beta  =  0.052980118f; /*    434 */
static const float dwt_gamma = -0.882911075f; /*  -7233 */
static const float dwt_delta = -0.443506852f; /*  -3633 */

static const float K      = 1.230174105f; /*  10078 */
static const float c13318 = 1.625732422f;

static opj_mqc_state_t mqc_states[47 * 2] = {
	{0x5601, 0, &mqc_states[2], &mqc_states[3]},
	{0x5601, 1, &mqc_states[3], &mqc_states[2]},
	{0x3401, 0, &mqc_states[4], &mqc_states[12]},
	{0x3401, 1, &mqc_states[5], &mqc_states[13]},
	{0x1801, 0, &mqc_states[6], &mqc_states[18]},
	{0x1801, 1, &mqc_states[7], &mqc_states[19]},
	{0x0ac1, 0, &mqc_states[8], &mqc_states[24]},
	{0x0ac1, 1, &mqc_states[9], &mqc_states[25]},
	{0x0521, 0, &mqc_states[10], &mqc_states[58]},
	{0x0521, 1, &mqc_states[11], &mqc_states[59]},
	{0x0221, 0, &mqc_states[76], &mqc_states[66]},
	{0x0221, 1, &mqc_states[77], &mqc_states[67]},
	{0x5601, 0, &mqc_states[14], &mqc_states[13]},
	{0x5601, 1, &mqc_states[15], &mqc_states[12]},
	{0x5401, 0, &mqc_states[16], &mqc_states[28]},
	{0x5401, 1, &mqc_states[17], &mqc_states[29]},
	{0x4801, 0, &mqc_states[18], &mqc_states[28]},
	{0x4801, 1, &mqc_states[19], &mqc_states[29]},
	{0x3801, 0, &mqc_states[20], &mqc_states[28]},
	{0x3801, 1, &mqc_states[21], &mqc_states[29]},
	{0x3001, 0, &mqc_states[22], &mqc_states[34]},
	{0x3001, 1, &mqc_states[23], &mqc_states[35]},
	{0x2401, 0, &mqc_states[24], &mqc_states[36]},
	{0x2401, 1, &mqc_states[25], &mqc_states[37]},
	{0x1c01, 0, &mqc_states[26], &mqc_states[40]},
	{0x1c01, 1, &mqc_states[27], &mqc_states[41]},
	{0x1601, 0, &mqc_states[58], &mqc_states[42]},
	{0x1601, 1, &mqc_states[59], &mqc_states[43]},
	{0x5601, 0, &mqc_states[30], &mqc_states[29]},
	{0x5601, 1, &mqc_states[31], &mqc_states[28]},
	{0x5401, 0, &mqc_states[32], &mqc_states[28]},
	{0x5401, 1, &mqc_states[33], &mqc_states[29]},
	{0x5101, 0, &mqc_states[34], &mqc_states[30]},
	{0x5101, 1, &mqc_states[35], &mqc_states[31]},
	{0x4801, 0, &mqc_states[36], &mqc_states[32]},
	{0x4801, 1, &mqc_states[37], &mqc_states[33]},
	{0x3801, 0, &mqc_states[38], &mqc_states[34]},
	{0x3801, 1, &mqc_states[39], &mqc_states[35]},
	{0x3401, 0, &mqc_states[40], &mqc_states[36]},
	{0x3401, 1, &mqc_states[41], &mqc_states[37]},
	{0x3001, 0, &mqc_states[42], &mqc_states[38]},
	{0x3001, 1, &mqc_states[43], &mqc_states[39]},
	{0x2801, 0, &mqc_states[44], &mqc_states[38]},
	{0x2801, 1, &mqc_states[45], &mqc_states[39]},
	{0x2401, 0, &mqc_states[46], &mqc_states[40]},
	{0x2401, 1, &mqc_states[47], &mqc_states[41]},
	{0x2201, 0, &mqc_states[48], &mqc_states[42]},
	{0x2201, 1, &mqc_states[49], &mqc_states[43]},
	{0x1c01, 0, &mqc_states[50], &mqc_states[44]},
	{0x1c01, 1, &mqc_states[51], &mqc_states[45]},
	{0x1801, 0, &mqc_states[52], &mqc_states[46]},
	{0x1801, 1, &mqc_states[53], &mqc_states[47]},
	{0x1601, 0, &mqc_states[54], &mqc_states[48]},
	{0x1601, 1, &mqc_states[55], &mqc_states[49]},
	{0x1401, 0, &mqc_states[56], &mqc_states[50]},
	{0x1401, 1, &mqc_states[57], &mqc_states[51]},
	{0x1201, 0, &mqc_states[58], &mqc_states[52]},
	{0x1201, 1, &mqc_states[59], &mqc_states[53]},
	{0x1101, 0, &mqc_states[60], &mqc_states[54]},
	{0x1101, 1, &mqc_states[61], &mqc_states[55]},
	{0x0ac1, 0, &mqc_states[62], &mqc_states[56]},
	{0x0ac1, 1, &mqc_states[63], &mqc_states[57]},
	{0x09c1, 0, &mqc_states[64], &mqc_states[58]},
	{0x09c1, 1, &mqc_states[65], &mqc_states[59]},
	{0x08a1, 0, &mqc_states[66], &mqc_states[60]},
	{0x08a1, 1, &mqc_states[67], &mqc_states[61]},
	{0x0521, 0, &mqc_states[68], &mqc_states[62]},
	{0x0521, 1, &mqc_states[69], &mqc_states[63]},
	{0x0441, 0, &mqc_states[70], &mqc_states[64]},
	{0x0441, 1, &mqc_states[71], &mqc_states[65]},
	{0x02a1, 0, &mqc_states[72], &mqc_states[66]},
	{0x02a1, 1, &mqc_states[73], &mqc_states[67]},
	{0x0221, 0, &mqc_states[74], &mqc_states[68]},
	{0x0221, 1, &mqc_states[75], &mqc_states[69]},
	{0x0141, 0, &mqc_states[76], &mqc_states[70]},
	{0x0141, 1, &mqc_states[77], &mqc_states[71]},
	{0x0111, 0, &mqc_states[78], &mqc_states[72]},
	{0x0111, 1, &mqc_states[79], &mqc_states[73]},
	{0x0085, 0, &mqc_states[80], &mqc_states[74]},
	{0x0085, 1, &mqc_states[81], &mqc_states[75]},
	{0x0049, 0, &mqc_states[82], &mqc_states[76]},
	{0x0049, 1, &mqc_states[83], &mqc_states[77]},
	{0x0025, 0, &mqc_states[84], &mqc_states[78]},
	{0x0025, 1, &mqc_states[85], &mqc_states[79]},
	{0x0015, 0, &mqc_states[86], &mqc_states[80]},
	{0x0015, 1, &mqc_states[87], &mqc_states[81]},
	{0x0009, 0, &mqc_states[88], &mqc_states[82]},
	{0x0009, 1, &mqc_states[89], &mqc_states[83]},
	{0x0005, 0, &mqc_states[90], &mqc_states[84]},
	{0x0005, 1, &mqc_states[91], &mqc_states[85]},
	{0x0001, 0, &mqc_states[90], &mqc_states[86]},
	{0x0001, 1, &mqc_states[91], &mqc_states[87]},
	{0x5601, 0, &mqc_states[92], &mqc_states[92]},
	{0x5601, 1, &mqc_states[93], &mqc_states[93]},
};


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
__global__ void kernel_v4dwt_h_global_cas1(float *d_tilec_data, int h_wavelet_sn, int h_wavelet_dn, int h_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rw, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta, float4 *d_wavelet) {

	unsigned int i = blockIdx.x;
	unsigned int j = threadIdx.x; 
	
	int offset = 4*i*rw; 
	int iterations = ceil(rw/((float)THRESHOLD_SHARED_DIM));
	int count = 0;
	
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			if(j < h_wavelet_sn) {
				d_wavelet[offset + ((2*j)+1)].x = const1 * d_tilec_data[i*4*w + j];
				d_wavelet[offset + ((2*j)+1)].y = const1 * d_tilec_data[(((4*i) + 1)*w) + j];
				d_wavelet[offset + ((2*j)+1)].z = const1 * d_tilec_data[(((4*i) + 2)*w) + j];
				d_wavelet[offset + ((2*j)+1)].w = const1 * d_tilec_data[(((4*i) + 3)*w) + j];    
			} else if(j < h_wavelet_sn + h_wavelet_dn){ 
				int p = j - h_wavelet_sn;
				d_wavelet[offset + p*2].x = const2 * d_tilec_data[i*4*w + j];
				d_wavelet[offset + p*2].y = const2 * d_tilec_data[(((4*i) + 1)*w) + j];
				d_wavelet[offset + p*2].z = const2 * d_tilec_data[(((4*i) + 2)*w) + j];
				d_wavelet[offset + p*2].w = const2 * d_tilec_data[(((4*i) + 3)*w) + j];     
			}
		}
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	int a = 1;
	int b = 0;

	int k, m;

	// start at 0/2 case 
	k = h_wavelet_sn; 
	m = h_wavelet_dn - a;
	j = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			if(j%2==1) {
				if(j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_delta);
				} else if(j < 2*k) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m]*(f4_dwt_delta + f4_dwt_delta));
				}
			}
		}
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();
	

	// start at 1/1 case 
	k = h_wavelet_dn;
	m = h_wavelet_sn - b;
	j = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			if(j%2 == 0) { 
				if(j > 0 && j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_gamma);
				} else if(j == 0){ 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j+1] + d_wavelet[offset+j+1])*f4_dwt_gamma);
				} else if(j < 2*k){
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m-1]*(f4_dwt_gamma + f4_dwt_gamma)); 
				}	
			}
		}
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	// start at 0/2 case 
	k = h_wavelet_sn; 
	m = h_wavelet_dn - a;
	j = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			if(j%2==1) {
				if(j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_beta);
				} else if(j < 2*k) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m]*(f4_dwt_beta + f4_dwt_beta));
				}
			}
		}
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();
	

	// start at 1/1 case 
	k = h_wavelet_dn;
	m = h_wavelet_sn - b;
	j = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			if(j%2 == 0) { 
				if(j > 0 && j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_alpha);
				} else if(j == 0){ 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j+1] + d_wavelet[offset+j+1])*f4_dwt_alpha);
				} else if(j < 2*k){
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m-1]*(f4_dwt_alpha + f4_dwt_alpha)); 
				}	
			}
		}
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();


	j = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			d_tilec_data[i*4*w + j] = d_wavelet[offset+j].x;
			d_tilec_data[(((4*i) + 1)*w) + j] = d_wavelet[offset+j].y;
			d_tilec_data[(((4*i) + 2)*w) + j] = d_wavelet[offset+j].z;
			d_tilec_data[(((4*i) + 3)*w) + j] = d_wavelet[offset+j].w;
		}
		j+=THRESHOLD_SHARED_DIM; 
	}
}

__global__ void kernel_v4dwt_h_global_cas0(float *d_tilec_data, int h_wavelet_sn, int h_wavelet_dn, int h_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rw, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta, float4 *d_wavelet) {

	unsigned int i = blockIdx.x;
	unsigned int j = threadIdx.x; 
	
	int offset = 4*i*rw; 
	int iterations = ceil(rw/((float)THRESHOLD_SHARED_DIM));
	int count = 0;
	
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			if(j < h_wavelet_sn) {
				d_wavelet[offset + 2*j].x = const1 * d_tilec_data[i*4*w + j];
				d_wavelet[offset + 2*j].y = const1 * d_tilec_data[(((4*i) + 1)*w) + j];
				d_wavelet[offset + 2*j].z = const1 * d_tilec_data[(((4*i) + 2)*w) + j];
				d_wavelet[offset + 2*j].w = const1 * d_tilec_data[(((4*i) + 3)*w) + j];    
			} else if(j < h_wavelet_sn + h_wavelet_dn){ 
				int p = j - h_wavelet_sn;
				d_wavelet[offset + (p*2 + 1)].x = const2 * d_tilec_data[i*4*w + j];
				d_wavelet[offset + (p*2 + 1)].y = const2 * d_tilec_data[(((4*i) + 1)*w) + j];
				d_wavelet[offset + (p*2 + 1)].z = const2 * d_tilec_data[(((4*i) + 2)*w) + j];
				d_wavelet[offset + (p*2 + 1)].w = const2 * d_tilec_data[(((4*i) + 3)*w) + j];     
			}
		}		
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	
	int a = 0;
	int b = 1;

	int k, m;

	// start at 1/1 case 
	k = h_wavelet_sn;
	m = h_wavelet_dn - a;
	j = threadIdx.x; 
	for(count =  0; count < iterations; count++) { 
		if(j < rw) { 
			if(j%2 == 0) { 
				if(j > 0 && j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_delta);
				} else if(j == 0){ 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j+1] + d_wavelet[offset+j+1])*f4_dwt_delta);
				} else if(j < 2*k){
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m-1]*(f4_dwt_delta + f4_dwt_delta)); 
				}
			}
		}
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	// start at 0/2 case 
	k = h_wavelet_dn; 
	m = h_wavelet_sn - b;
	j = threadIdx.x;
	for(count = 0; count < iterations; count++) { 
		if(j < rw) { 
			if(j%2==1) {
				if(j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_gamma);
				} else if(j < 2*k) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m]*(f4_dwt_gamma + f4_dwt_gamma));
				}
			}
		}
		j+=THRESHOLD_SHARED_DIM; 
	}

	__syncthreads();
	
	// start at 1/1 case 
	k = h_wavelet_sn;
	m = h_wavelet_dn - a;
	j = threadIdx.x;
	for(count =  0; count < iterations; count++) { 
		if(j < rw) { 
			if(j%2 == 0) { 
				if(j > 0 && j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_beta);
				} else if(j == 0){ 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j+1] + d_wavelet[offset+j+1])*f4_dwt_beta);
				} else if(j < 2*k){
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m-1]*(f4_dwt_beta + f4_dwt_beta)); 
				}
			}
		}
		j+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	// start at 0/2 case 
	k = h_wavelet_dn; 
	m = h_wavelet_sn - b;
	j = threadIdx.x;
	for(count = 0; count < iterations; count++) { 
		if(j < rw) { 
			if(j%2==1) {
				if(j < 2*m) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + ((d_wavelet[offset+j-1] + d_wavelet[offset+j+1])*f4_dwt_alpha);
				} else if(j < 2*k) { 
					d_wavelet[offset+j] = d_wavelet[offset+j] + (d_wavelet[2*m]*(f4_dwt_alpha + f4_dwt_alpha));
				}
			}
		}
		j+=THRESHOLD_SHARED_DIM; 
	}
	__syncthreads();
	

	j = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(j < rw) { 
			d_tilec_data[i*4*w + j] = d_wavelet[offset+j].x;
			d_tilec_data[(((4*i) + 1)*w) + j] = d_wavelet[offset+j].y;
			d_tilec_data[(((4*i) + 2)*w) + j] = d_wavelet[offset+j].z;
			d_tilec_data[(((4*i) + 3)*w) + j] = d_wavelet[offset+j].w;
		}
		j+=THRESHOLD_SHARED_DIM; 
	}
}
__global__ void kernel_v4dwt_h_cas1(float *d_tilec_data, int h_wavelet_sn, int h_wavelet_dn, int h_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rw, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta) {

	unsigned int i = blockIdx.x;
	unsigned int j = threadIdx.x; 

	__shared__ float4 shared_h_wavelet[THRESHOLD_SHARED_DIM];
		
	__syncthreads();  

	if(j < h_wavelet_sn) {
		shared_h_wavelet[((2*j)+1)].x = const1 * d_tilec_data[i*4*w + j];
		shared_h_wavelet[((2*j)+1)].y = const1 * d_tilec_data[(((4*i) + 1)*w) + j];
		shared_h_wavelet[((2*j)+1)].z = const1 * d_tilec_data[(((4*i) + 2)*w) + j];
		shared_h_wavelet[((2*j)+1)].w = const1 * d_tilec_data[(((4*i) + 3)*w) + j];    
	} else if(j < h_wavelet_sn + h_wavelet_dn){ 
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

	__shared__ float4 shared_h_wavelet[THRESHOLD_SHARED_DIM];
		
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

	// start at 1/1 case 
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

	// start at 0/2 case 
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

	// start at 1/1 case
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

	// start at 0/2 case 
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

__global__ void kernel_v4dwt_v_global_cas1(float *d_tilec_data, int v_wavelet_sn, int v_wavelet_dn, int v_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rh, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta, float4 *d_wavelet) {

	unsigned int i = threadIdx.x; // 0 to rh
	unsigned int j = blockIdx.x; // 0 to rw/4

	int offset = 4*j*rh; 
	int iterations = ceil(rh/((float)THRESHOLD_SHARED_DIM));
	int count = 0;
	
	for(count = 0; count < iterations; count++) {
		if(i < rh)  { 
			if(i < v_wavelet_sn) {
				d_wavelet[offset + ((2*i)+1)].x = const1 * d_tilec_data[i*w + (4*j)];
				d_wavelet[offset + ((2*i)+1)].y = const1 * d_tilec_data[i*w + (4*j) + 1];
				d_wavelet[offset + ((2*i)+1)].z = const1 * d_tilec_data[i*w + (4*j) + 2];
				d_wavelet[offset + ((2*i)+1)].w = const1 * d_tilec_data[i*w + (4*j) + 3];    
			} else { 
				int p = i - v_wavelet_sn;
				d_wavelet[offset + p*2].x = const2 * d_tilec_data[i*w + (4*j)];
				d_wavelet[offset + p*2].y = const2 * d_tilec_data[i*w + (4*j) + 1];
				d_wavelet[offset + p*2].z = const2 * d_tilec_data[i*w + (4*j) + 2];
				d_wavelet[offset + p*2].w = const2 * d_tilec_data[i*w + (4*j) + 3];     
			}
		}
		i+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	int a = 1;
	int b = 0;

	int k, m;

	// start at 0/2 case
	k = v_wavelet_sn;
	m = v_wavelet_dn - a;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) { 
		if(i < rh) { 
			if(i%2==1) {
				if(i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_delta);
				} else if(i < 2*k) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m]*(f4_dwt_delta + f4_dwt_delta));
				}
			}
		}
		i+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	
	// start at 1/1 case 
	k = v_wavelet_dn;
	m = v_wavelet_sn - b;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) { 
		if(i < rh) { 
			if(i%2 == 0) { 
				if(i > 0 && i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_gamma);
				} else if(i == 0){ 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i+1] + d_wavelet[offset+i+1])*f4_dwt_gamma);
				} else if(i < 2*k){
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m-1]*(f4_dwt_gamma + f4_dwt_gamma)); 
				}
			}
		}
		i+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	// start at 0/2 case
	k = v_wavelet_sn;
	m = v_wavelet_dn - a;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) { 
		if(i < rh) { 
			if(i%2==1) {
				if(i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_beta);
				} else if(i < 2*k) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m]*(f4_dwt_beta + f4_dwt_beta));
				}
			}
		}
		i+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	
	// start at 1/1 case 
	k = v_wavelet_dn;
	m = v_wavelet_sn - b;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) { 
		if(i < rh) { 
			if(i%2 == 0) { 
				if(i > 0 && i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_alpha);
				} else if(i == 0){ 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i+1] + d_wavelet[offset+i+1])*f4_dwt_alpha);
				} else if(i < 2*k){
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m-1]*(f4_dwt_alpha + f4_dwt_alpha)); 
				}
			}
		}
		i+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	i = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(i < rh) { 
			d_tilec_data[i*w + (4*j)] = d_wavelet[offset+i].x;
			d_tilec_data[i*w + (4*j) + 1] = d_wavelet[offset+i].y;
			d_tilec_data[i*w + (4*j) + 2] = d_wavelet[offset+i].z;
			d_tilec_data[i*w + (4*j) + 3] = d_wavelet[offset+i].w; 
		}
		i+=THRESHOLD_SHARED_DIM;
	}
}

__global__ void kernel_v4dwt_v_global_cas0(float *d_tilec_data, int v_wavelet_sn, int v_wavelet_dn, int v_wavelet_cas, unsigned int w, unsigned int buffsize, unsigned int rh, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta, float4 *d_wavelet) {

	unsigned int i = threadIdx.x; // 0 to rh
	unsigned int j = blockIdx.x; // 0 to rw/4

 	int offset = 4*j*rh; 
	int iterations = ceil(rh/((float)THRESHOLD_SHARED_DIM));
	int count = 0;
	
	for(count = 0; count < iterations; count++) {
		if(i < rh)  { 
			if(i < v_wavelet_sn) {
				d_wavelet[offset + 2*i].x = const1 * d_tilec_data[i*w + (4*j)];
				d_wavelet[offset + 2*i].y = const1 * d_tilec_data[i*w + (4*j) + 1];
				d_wavelet[offset + 2*i].z = const1 * d_tilec_data[i*w + (4*j) + 2];
				d_wavelet[offset + 2*i].w = const1 * d_tilec_data[i*w + (4*j) + 3];    
			} else { 
				int p = i - v_wavelet_sn;
				d_wavelet[offset + (p*2 + 1)].x = const2 * d_tilec_data[i*w + (4*j)];
				d_wavelet[offset + (p*2 + 1)].y = const2 * d_tilec_data[i*w + (4*j) + 1];
				d_wavelet[offset + (p*2 + 1)].z = const2 * d_tilec_data[i*w + (4*j) + 2];
				d_wavelet[offset + (p*2 + 1)].w = const2 * d_tilec_data[i*w + (4*j) + 3];     
			}
		}
		i+=THRESHOLD_SHARED_DIM; 
	}
	__syncthreads();

	int a = 0;
	int b = 1;

	int k, m;


	/* start at 1/1 case */
	k = v_wavelet_sn;
	m = v_wavelet_dn - a;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(i < rh)  { 
			if(i%2 == 0) { 
				if(i > 0 && i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_delta);
				} else if(i == 0){ 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i+1] + d_wavelet[offset+i+1])*f4_dwt_delta);
				} else if(i < 2*k){
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m-1]*(f4_dwt_delta + f4_dwt_delta)); 
				}		
			}
		}
		i+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	/* start at 0/2 case */
	k = v_wavelet_dn; 
	m = v_wavelet_sn - b;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(i < rh)  { 
			if(i%2==1) {
				if(i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_gamma);
				} else if(i < 2*k) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m]*(f4_dwt_gamma + f4_dwt_gamma));
				}
			}
		}
		i+=THRESHOLD_SHARED_DIM; 
	}
	__syncthreads();


	/* start at 1/1 case */
	k = v_wavelet_sn;
	m = v_wavelet_dn - a;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(i < rh)  { 
			if(i%2 == 0) { 
				if(i > 0 && i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_beta);
				} else if(i == 0){ 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i+1] + d_wavelet[offset+i+1])*f4_dwt_beta);
				} else if(i < 2*k){
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m-1]*(f4_dwt_beta + f4_dwt_beta)); 
				}		
			}
		}
		i+=THRESHOLD_SHARED_DIM;
	}
	__syncthreads();

	/* start at 0/2 case */
	k = v_wavelet_dn; 
	m = v_wavelet_sn - b;
	i = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(i < rh)  { 
			if(i%2==1) {
				if(i < 2*m) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + ((d_wavelet[offset+i-1] + d_wavelet[offset+i+1])*f4_dwt_alpha);
				} else if(i < 2*k) { 
					d_wavelet[offset+i] = d_wavelet[offset+i] + (d_wavelet[2*m]*(f4_dwt_alpha + f4_dwt_alpha));
				}
			}
		}
		i+=THRESHOLD_SHARED_DIM; 
	}
	__syncthreads();

	i = threadIdx.x;
	for(count = 0; count < iterations; count++) {
		if(i < rh)  { 
			d_tilec_data[i*w + (4*j)] = d_wavelet[offset+i].x;
			d_tilec_data[i*w + (4*j) + 1] = d_wavelet[offset+i].y;
			d_tilec_data[i*w + (4*j) + 2] = d_wavelet[offset+i].z;
			d_tilec_data[i*w + (4*j) + 3] = d_wavelet[offset+i].w; 
		}
		i+=THRESHOLD_SHARED_DIM;
	}
}


__global__ void kernel_v4dwt_v_cas1(float *d_tilec_data, int v_wavelet_sn, int v_wavelet_dn, int v_wavelet_cas, unsigned int w, unsigned int buffsize, const float const1, const float const2, float4 f4_dwt_alpha, float4 f4_dwt_beta, float4 f4_dwt_gamma, float4 f4_dwt_delta) {

	unsigned int i = threadIdx.x; // 0 to rh
	unsigned int j = blockIdx.x; // 0 to rw/4

	__shared__ float4 shared_v_wavelet[THRESHOLD_SHARED_DIM];
		
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

	__shared__ float4 shared_v_wavelet[THRESHOLD_SHARED_DIM];
		
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

	int wavelet_size = dwt_max_wavelet_size_v2(res, numres) + 5;

	/* DEBUG
	float *result = (float *)opj_malloc(sizeof(float)*bufsize); */
	
	OPJ_FLOAT32 *d_tilec_data;
	cudaMalloc((OPJ_FLOAT32 **)&d_tilec_data, sizeof(OPJ_FLOAT32)*bufsize);
	cudaMemcpy(d_tilec_data, aj, sizeof(OPJ_FLOAT32)*bufsize, cudaMemcpyHostToDevice);

	float4 *d_wavelet; 
	if(wavelet_size > THRESHOLD_SHARED_DIM) { 
		cudaMalloc((float4 **)&d_wavelet, sizeof(float4)*wavelet_size*wavelet_size);
	}

	int h_wavelet_sn, h_wavelet_dn, h_wavelet_cas;
	int v_wavelet_sn, v_wavelet_dn, v_wavelet_cas;

	while( --numres) {

		h_wavelet_sn = rw;
		v_wavelet_sn = rh;

		++res;

		rw = res->x1 - res->x0;	/* width of the resolution level computed */
		rh = res->y1 - res->y0;	/* height of the resolution level computed */

		/* DEBUG
		printf("[GPU_DEBUG] rw %d, rh %d\n",rw,rh); */

		h_wavelet_dn = rw - h_wavelet_sn;
		h_wavelet_cas = res->x0 % 2;
		
		float4 f4_dwt_alpha = make_float4(dwt_alpha, dwt_alpha, dwt_alpha, dwt_alpha);
		float4 f4_dwt_beta = make_float4(dwt_beta, dwt_beta, dwt_beta, dwt_beta);
		float4 f4_dwt_gamma = make_float4(dwt_gamma, dwt_gamma, dwt_gamma, dwt_gamma);
		float4 f4_dwt_delta = make_float4(dwt_delta, dwt_delta, dwt_delta, dwt_delta);
		
	
		cudaThreadSynchronize();
		if(rw < THRESHOLD_SHARED_DIM) { 
			dim3 threads_h(rw, 1, 1);
			dim3 grid_h(ceil(rh/4.0), 1, 1);
			
			if(h_wavelet_cas == 0) { 
				kernel_v4dwt_h_cas0<<<grid_h, threads_h, 0>>>(d_tilec_data, h_wavelet_sn, h_wavelet_dn, h_wavelet_cas, w, bufsize, rw, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);
			} else { 
				kernel_v4dwt_h_cas1<<<grid_h, threads_h, 0>>>(d_tilec_data, h_wavelet_sn, h_wavelet_dn, h_wavelet_cas, w, bufsize, rw, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);
	
			}
		} else {
			dim3 threads_h(THRESHOLD_SHARED_DIM, 1, 1);
			dim3 grid_h(ceil(rh/4.0), 1, 1);
			if(h_wavelet_cas == 0) {
				kernel_v4dwt_h_global_cas0<<<grid_h, threads_h, 0>>>(d_tilec_data, h_wavelet_sn, h_wavelet_dn, h_wavelet_cas, w, bufsize, rw, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta, d_wavelet);
			} else { 
				kernel_v4dwt_h_global_cas1<<<grid_h, threads_h, 0>>>(d_tilec_data, h_wavelet_sn, h_wavelet_dn, h_wavelet_cas, w, bufsize, rw, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta, d_wavelet);
			}
		}
		cudaThreadSynchronize();

		/* DEBUG
		cudaError_t errorH = cudaGetLastError();  
		printf("[GPU_DEBUG] CUDA ERROR : %s\n",cudaGetErrorString(errorH)); */

		
		/* DEBUG v4dwt_h 
		OPJ_INT32 j,k;
		cudaMemcpy(result, d_tilec_data, sizeof(OPJ_FLOAT32)*bufsize, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		if(rw >= 512) { 
		for(j = rh; j > 3; j -= 4) {
			for(k = rw; --k >= 0;){
				printf("rh %d rw %d (%d,%d,%d,%d)\n",j,k,(int)(floor(result[k])),(int)(floor(result[k+w])),(int)(floor(result[k+w*2])),(int)(floor(result[k+w*3])));
			}
			result+=w*4;
		}
		printf("NEXT ITERATION\n");
		} */

		v_wavelet_dn = rh - v_wavelet_sn;
		v_wavelet_cas = res->y0 % 2;

		dim3 threads_v(rh, 1, 1);
		dim3 grid_v(ceil(rw/4.0),1,1);

		
		cudaThreadSynchronize();
		if(rh < THRESHOLD_SHARED_DIM) { 
			dim3 threads_v(rh, 1, 1);
			dim3 grid_v(ceil(rw/4.0),1,1);
			if(v_wavelet_cas == 0) { 
				kernel_v4dwt_v_cas0<<<grid_v, threads_v, 0>>>(d_tilec_data, v_wavelet_sn, v_wavelet_dn, v_wavelet_cas, w, bufsize, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);
			} else { 
				kernel_v4dwt_v_cas1<<<grid_v, threads_v, 0>>>(d_tilec_data, v_wavelet_sn, v_wavelet_dn, v_wavelet_cas, w, bufsize, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta);
			}
		} else { 
			dim3 threads_v(THRESHOLD_SHARED_DIM, 1, 1);
			dim3 grid_v(ceil(rw/4.0),1,1);
			if(v_wavelet_cas == 0) { 
				kernel_v4dwt_v_global_cas0<<<grid_v, threads_v, 0>>>(d_tilec_data, v_wavelet_sn, v_wavelet_dn, v_wavelet_cas, w, bufsize, rh, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta, d_wavelet);
			} else { 
				kernel_v4dwt_v_global_cas1<<<grid_v, threads_v, 0>>>(d_tilec_data, v_wavelet_sn, v_wavelet_dn, v_wavelet_cas, w, bufsize, rh, K, c13318, f4_dwt_alpha, f4_dwt_beta, f4_dwt_gamma, f4_dwt_delta, d_wavelet);
	
			}
		}
		cudaThreadSynchronize();

		/* DEBUG
		cudaError_t errorV = cudaGetLastError();  
		printf("[GPU_DEBUG] CUDA ERROR : %s\n", cudaGetErrorString(errorV)); */

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

	cudaMemcpy(aj, d_tilec_data, sizeof(OPJ_FLOAT32)*bufsize, cudaMemcpyDeviceToHost);
	cudaFree(d_tilec_data);
	cudaFree(d_wavelet);
	return OPJ_TRUE;
}

__device__ static void device_func_t1_decode_cblk_v2(
		OPJ_UINT32 currentIndex,
		opj_t1_t *t1,
		opj_tcd_cblk_dec_v2_t* cblk,
		OPJ_UINT32 orient,
		OPJ_UINT32 roishift,
		OPJ_UINT32 cblksty,
		opj_mqc_state_t* mqc_states,
		opj_mqc_t* mqc,
		opj_tcd_seg_t* segs)
{

	OPJ_INT32 bpno;
	OPJ_UINT32 passtype;
	OPJ_UINT32 segno, passno;
	OPJ_BYTE type = T1_TYPE_MQ; 
	
	bpno = roishift + cblk->numbps - 1;
	passtype = 2;

	OPJ_INT32 i = 0;

	/* mqc_resetstates */
	for(i = 0; i < MQC_NUMCTXS; i++) { 
		mqc->ctxs[i] = mqc_states;
	}

	/* mqc_setstate */
	mqc->ctxs[T1_CTXNO_UNI] = &mqc_states[0 + (46 << 1)];
	mqc->ctxs[T1_CTXNO_AGG] = &mqc_states[0 + (3 << 1)];
	mqc->ctxs[T1_CTXNO_ZC] = &mqc_states[0 + (4 << 1)];

	// TODO: ADD segno and passno loop
} 

__global__ void kernel_t1_decode_cblks(opj_tcd_cblk_dec_v2_t* d_cblks_struct, unsigned int* len, 
		opj_tcd_band_v2_t* band, opj_tcd_resolution_v2_t* pres, opj_t1_t* t1, int *d_t1_data,
		opj_tccp_t* tccp, opj_mqc_state_t* mqc_states, opj_mqc_t* d_t1_mqc, opj_tcd_seg_t *d_t1_segs) {

	int currentIndex = blockIdx.x; 
	int threadID = threadIdx.x;

	unsigned int cblk_w, cblk_h;
	int x, y;
	unsigned int i = threadIdx.x;
	unsigned int j = threadIdx.y;

	int* restrict datap = d_t1_data;
	
	if(threadID == 0) {
		
		// calling decode one code block function here
		device_func_t1_decode_cblk_v2(currentIndex, t1, &d_cblks_struct[currentIndex], band->bandno, tccp->roishift, tccp->cblksty,
				mqc_states, &d_t1_mqc[currentIndex], d_t1_segs);
	
		x = d_cblks_struct[currentIndex].x0 - band->x0;
		y = d_cblks_struct[currentIndex].y0 - band->y0;
		
		if (band->bandno & 1) {
			x += pres->x1 - pres->x0;
		}
		if (band->bandno & 2) {
			y += pres->y1 - pres->y0;
		}
		
		cblk_w = d_cblks_struct[currentIndex].x1 - d_cblks_struct[currentIndex].x0;
		cblk_h = d_cblks_struct[currentIndex].y1 - d_cblks_struct[currentIndex].y0;

		len[currentIndex] = (&d_t1_segs[currentIndex*2*J2K_DEFAULT_NB_SEGS])->len; /* DEBUG */
	}

	//TODO : handle tccp->roishift 

	// parallel data filling part
	if( i < cblk_w && j < cblk_h) { 
		float tmp = datap[(j * cblk_w) + i] * band->stepsize;
	}

	return;
}

void gpu_t1_decode_cblks_across_cblkno(opj_t1_t* t1, opj_tcd_tilecomp_v2_t* tilec, opj_tccp_t* tccp, 
		opj_tcd_precinct_v2_t* precinct, opj_tcd_band_v2_t* band, OPJ_UINT32 resno, OPJ_UINT32 tile_w) {

	OPJ_INT32 cblkno = 0;
	OPJ_INT32 size = precinct->cw * precinct->ch;

	opj_tcd_cblk_dec_v2_t* d_cblks_struct;
	opj_tcd_band_v2_t* d_band;
	opj_t1_t* d_t1;
	opj_tccp_t* d_tccp;
	opj_mqc_state_t* d_mqc_states;

	unsigned int* d_len; 
	
	unsigned int* h_len = (unsigned int*)opj_malloc(sizeof(unsigned int)*size);

	dim3 threads(MAX_CBLOCK_HEIGHT, MAX_CBLOCK_WIDTH, 1);

	/* Memory allocation and initialization on GPU of OPJ Structs */
	cudaMalloc((opj_tcd_cblk_dec_v2_t** )&d_cblks_struct, sizeof(opj_tcd_cblk_dec_v2_t)*size);
	cudaMemcpy(d_cblks_struct, &precinct->cblks.dec[0], sizeof(opj_tcd_cblk_dec_v2_t)*size, cudaMemcpyHostToDevice);

	cudaMalloc((opj_tcd_band_v2_t** )&d_band, sizeof(opj_tcd_band_v2_t));
	cudaMemcpy(d_band, band, sizeof(opj_tcd_band_v2_t), cudaMemcpyHostToDevice);
	
	cudaMalloc((opj_t1_t** )&d_t1, sizeof(opj_t1_t));
	cudaMemcpy(d_t1, t1, sizeof(opj_t1_t), cudaMemcpyHostToDevice);

	cudaMalloc((opj_tccp_t** )&d_tccp, sizeof(opj_tccp_t));
	cudaMemcpy(d_tccp, tccp, sizeof(opj_tccp_t), cudaMemcpyHostToDevice);
	
	cudaMalloc((opj_mqc_state_t** )&d_mqc_states, sizeof(opj_mqc_state_t)*47*2);
	cudaMemcpy(d_mqc_states, mqc_states, sizeof(opj_mqc_state_t)*47*2, cudaMemcpyHostToDevice);
	
	/* Test array of size equal to number of code blocks to verify memory sanity */
	cudaMalloc((unsigned int** )&d_len, sizeof(unsigned int)*size);

	/* Memory allocation and/or initialization on GPU of arrays inside OPJ Structs */
	int d_t1_data_size = MAX_CBLOCK_HEIGHT*MAX_CBLOCK_WIDTH*size;
	int *d_t1_data;
	cudaMalloc((int**)&d_t1_data, sizeof(int)*d_t1_data_size);
	
	int d_t1_flags_size = (MAX_CBLOCK_WIDTH+2)*(MAX_CBLOCK_HEIGHT+2)*size;  
	flag_t *d_t1_flags; 
	cudaMalloc((flag_t**)&d_t1_flags, sizeof(flag_t)*d_t1_flags_size);

	int d_t1_mqc_size = size; 
	opj_mqc_t* d_t1_mqc;
	cudaMalloc((opj_mqc_t**)&d_t1_mqc, sizeof(opj_mqc_t)*d_t1_mqc_size);

	int d_t1_segs_size = 2*J2K_DEFAULT_NB_SEGS*size;
	opj_tcd_seg_t* d_t1_segs;
	cudaMalloc((opj_tcd_seg_t**)&d_t1_segs, sizeof(opj_tcd_seg_t)*d_t1_segs_size);

	for(cblkno = 0; cblkno < size; cblkno++) { 
		opj_tcd_cblk_dec_v2_t* cblk = &precinct->cblks.dec[cblkno];
		cudaMemcpy(d_t1_segs + (2*J2K_DEFAULT_NB_SEGS*cblkno), cblk->segs, sizeof(opj_tcd_seg_t)*cblk->real_num_segs, cudaMemcpyHostToDevice);
	}
	
	opj_tcd_resolution_v2_t* pres;
	opj_tcd_resolution_v2_t* d_pres;
	if(band->bandno & 1 || band->bandno & 2) { 
	 	pres = &tilec->resolutions[resno - 1];
		cudaMalloc((opj_tcd_resolution_v2_t** )&d_pres, sizeof(opj_tcd_resolution_v2_t));
		cudaMemcpy(d_pres, pres, sizeof(opj_tcd_resolution_v2), cudaMemcpyHostToDevice);
	} 
	
	// DEBUG printf("Calling Kernel with data size %d\n",t1->datasize);

	kernel_t1_decode_cblks<<<size, threads, 0>>>(d_cblks_struct, d_len, d_band, d_pres, d_t1, d_t1_data, d_tccp,
			d_mqc_states, d_t1_mqc, d_t1_segs);

	cudaMemcpy(h_len, d_len, sizeof(unsigned int)*size, cudaMemcpyDeviceToHost);

	cudaError_t errorV = cudaGetLastError();  
	printf("[GPU_T1_DECODE_DEBUG] CUDA ERROR : %s\n", cudaGetErrorString(errorV)); 

	/* DEBUG
	for(cblkno = 0; cblkno < size; cblkno++) {
		opj_tcd_cblk_dec_v2_t* cblk = &precinct->cblks.dec[cblkno];
		int x = cblk->x0 - band->x0;
		int y = cblk->y0 - band->y0;
		if (band->bandno & 1) {
			opj_tcd_resolution_v2_t* pres = &tilec->resolutions[resno - 1];
			x += pres->x1 - pres->x0;
		}
		if (band->bandno & 2) {
			opj_tcd_resolution_v2_t* pres = &tilec->resolutions[resno - 1];
			y += pres->y1 - pres->y0;
		}

		unsigned int cblk_w = t1->w;
		unsigned int cblk_h = t1->h;
		printf("GPU data[%d] = %u, corresponding CPU data = %u, %u\n", cblkno, h_len[cblkno], (&cblk->segs[0])->len, cblk->real_num_segs);
	}  */
}

