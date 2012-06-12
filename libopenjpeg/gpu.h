
#ifndef __GPU_H
#define __GPU_H

#define MAX_THREADS_PER_BLOCK 512

/**
Perform Inverse DC Level Shift on the GPU
@param p_tcd_v2 TCD Handle
*/

opj_bool gpu_dc_level_shift_decode( opj_tcd_v2_t *p_tcd);

opj_bool gpu_dwt_decode_real_v2(opj_tcd_tilecomp_v2_t* restrict tilec, OPJ_UINT32 numres);

#endif
