/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include <stdlib.h>
#include <stdio.h>
#include "support.h"

// These constants valid for the IEEE 494 bus interconnect matrix
#ifndef spmv_H__
#define spmv_H__
#define NNZ 1666
#define N_MAT 494

#define TYPE double
#endif

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N_MAT + 1], TYPE vec[N_MAT], TYPE out[N_MAT]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE val[NNZ];
  int32_t cols[NNZ];
  int32_t rowDelimiters[N_MAT+1];
  TYPE vec[N_MAT];
  TYPE out[N_MAT];
};
