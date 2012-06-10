
#ifndef __GPU_H
#define __GPU_H

#define MAX_THREADS_PER_BLOCK 512

/**
Perform Inverse DC Level Shift on the GPU
@param p_tcd_v2 TCD Handle
*/

opj_bool gpu_dc_level_shift_decode( opj_tcd_v2_t *p_tcd);

#endif
