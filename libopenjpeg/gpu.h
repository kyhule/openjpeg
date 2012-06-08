
#ifndef __GPU_H
#define __GPU_H

#define MAX_THREADS_PER_BLOCK 512

/**
Perform Inverse DC Level Shift on the GPU
@param current_ptr tile data
@param m_dc_level_shift value to be added to tile data
@param height tile width
@param width  tile height
*/
void gpu_dc_level_shift_decode( OPJ_INT32 *current_ptr, 
		 	        OPJ_INT32 m_dc_level_shift,  
			        OPJ_INT32 min, OPJ_INT32 max, 
			        OPJ_UINT32 stride, 
			        OPJ_UINT32 height, 
			        OPJ_UINT32 width, 
			        OPJ_INT32 qmfbid );

#endif
