
#ifndef __GPU_H
#define __GPU_H

#define MAX_THREADS_PER_BLOCK 512
#define MAX_THREADS_HEIGHT 16
#define MAX_THREADS_WIDTH 16
#define THRESHOLD_SHARED_DIM 512
#define MAX_CBLOCK_HEIGHT 32
#define MAX_CBLOCK_WIDTH 16

/**
Perform Inverse DC Level Shift on the GPU
@param p_tcd_v2 TCD Handle
*/

opj_bool gpu_dc_level_shift_decode(opj_tcd_v2_t *p_tcd);

opj_bool gpu_dwt_decode_real_v2(opj_tcd_tilecomp_v2_t* restrict tilec, OPJ_UINT32 numres);

void gpu_t1_decode_cblks_across_cblkno( opj_t1_t* t, opj_tcd_tilecomp_v2_t* tilec, opj_tccp_t* tccp, 
		opj_tcd_precinct_v2_t* precinct, opj_tcd_band_v2_t* band, OPJ_UINT32 resno, OPJ_UINT32 tile_w);

#endif
