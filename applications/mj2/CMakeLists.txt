# Makefile for the MJ2 codecs of the OpenJPEG library: frames_to_mj2, mj2_to_frames, extract_j2k_from_mj2 and wrap_j2k_in_mj2

SET(common_SRCS ${OPENJPEG_SOURCE_DIR}/applications/common/opj_getopt.c)

# While mj2 executables do not use the API correctly, we do not link with the library but rather compile the sources files.
SET(OPJ_SRCS
${OPENJPEG_SOURCE_DIR}/libopenjpeg/bio.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/cio.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/dwt.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/event.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/gpu.cu
${OPENJPEG_SOURCE_DIR}/libopenjpeg/image.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/j2k.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/j2k_lib.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/jp2.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/jpt.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/mct.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/mqc.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/openjpeg.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/pi.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/raw.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/t1.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/t2.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/tcd.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/tgt.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/cidx_manager.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/phix_manager.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/ppix_manager.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/thix_manager.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/tpix_manager.c
${OPENJPEG_SOURCE_DIR}/libopenjpeg/function_list.c
)

SET(EXEC_MJ2_SRCS
${OPENJPEG_SOURCE_DIR}/applications/mj2/frames_to_mj2.c
${OPENJPEG_SOURCE_DIR}/applications/mj2/mj2_to_frames.c
${OPENJPEG_SOURCE_DIR}/applications/mj2/wrap_j2k_in_mj2.c
${OPENJPEG_SOURCE_DIR}/applications/mj2/extract_j2k_from_mj2.c
)

SET(MJ2_SRCS mj2.c mj2_convert.c)

set_source_files_properties(${OPJ_SRCS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${MJ2_SRCS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${common_SRCS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${EXEC_MJ2_SRCS} PROPERTIES LANGUAGE CXX)


INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRECTORIES})

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)


IF(WIN32)
  ADD_DEFINITIONS(-DOPJ_STATIC)
ENDIF(WIN32)

# Headers file are located here:
INCLUDE_DIRECTORIES(
  ${OPENJPEG_SOURCE_DIR}/libopenjpeg
  ${OPENJPEG_SOURCE_DIR}/applications/common
  ${LCMS_INCLUDE_DIRNAME}
  )

CUDA_ADD_EXECUTABLE(frames_to_mj2
  frames_to_mj2.c
  ${common_SRCS}
  ${OPJ_SRCS}
  ${MJ2_SRCS}
  )
TARGET_LINK_LIBRARIES(frames_to_mj2 ${LCMS_LIBNAME} ${CUDA_LIBRARIES})

IF(UNIX)
  TARGET_LINK_LIBRARIES(frames_to_mj2 m)
ENDIF(UNIX)

CUDA_ADD_EXECUTABLE(mj2_to_frames
    mj2_to_frames.c
    ${common_SRCS}
    ${OPJ_SRCS}
    ${MJ2_SRCS}
    ${OPENJPEG_SOURCE_DIR}/applications/common/color.c
    )
TARGET_LINK_LIBRARIES(mj2_to_frames ${LCMS_LIBNAME} ${CUDA_LIBRARIES})

IF(UNIX)
  TARGET_LINK_LIBRARIES(mj2_to_frames m)
ENDIF(UNIX)

CUDA_ADD_EXECUTABLE(extract_j2k_from_mj2
    extract_j2k_from_mj2.c
    ${OPJ_SRCS}
    ${MJ2_SRCS}
    )
TARGET_LINK_LIBRARIES(extract_j2k_from_mj2 ${LCMS_LIBNAME} ${CUDA_LIBRARIES})

IF(UNIX)
  TARGET_LINK_LIBRARIES(extract_j2k_from_mj2 m ${CUDA_LIBRARIES})
ENDIF(UNIX)

CUDA_ADD_EXECUTABLE(wrap_j2k_in_mj2
    wrap_j2k_in_mj2.c
    ${OPJ_SRCS}
    ${MJ2_SRCS}
    )
TARGET_LINK_LIBRARIES(wrap_j2k_in_mj2 ${LCMS_LIBNAME} ${CUDA_LIBRARIES})

IF(UNIX)
  TARGET_LINK_LIBRARIES(wrap_j2k_in_mj2 m)
ENDIF(UNIX)

INSTALL(TARGETS frames_to_mj2 mj2_to_frames extract_j2k_from_mj2 wrap_j2k_in_mj2
	DESTINATION ${OPENJPEG_INSTALL_BIN_DIR})
