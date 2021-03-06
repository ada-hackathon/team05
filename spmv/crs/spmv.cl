/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description: 
    OpenCL Dataflow Example using xcl_dataflow attribute 
    This is example of vector addition to demonstrate OpenCL Dataflow xcl_dataflow 
    functionality to perform task/function level parallelism using xcl_dataflow
    attribute. OpenCL xcl_dataflow instruct compiler to run functions inside kernel 
    concurrently. In this Example, a vector addition implementation is divided into 
    three sub-task APIs as below:
    1) read_input(): 
        This API reads the input vector from Global Memory and writes it into 
        'buffer_in'.
    2) compute_add(): 
        This API reads the input vector from 'buffer_in' and increment the value 
        by user specified increment. It writes the result into 'buffer_out'.
    3) write_result(): 
        This API reads the result vector from 'buffer_out' and write the result 
        into Global Memory Location.
    Data Flow based Adder will be implemented as below:
                    _____________
                    |             |<----- Input Vector from Global Memory
                    |  read_input |       __
                    |_____________|----->|  |
                     _____________       |  | buffer_in
                    |             |<-----|__|
                    | compute_add |       __
                    |_____________|----->|  |
                     _____________       |  | buffer_out
                    |              |<----|__|
                    | write_result |       
                    |______________|-----> Output result to Global Memory
*******************************************************************************/
#define NNZ 1666
#define N_MAT 494

#define TYPE double
#define BUFFER_SIZE 4096

// Read Input data from buffer_in and write the result into buffer_out
static void compute_spmv(__global TYPE val[NNZ], __global int cols[NNZ], __global int rowDelimiters[N_MAT+1], __global TYPE vec[N_MAT], __global TYPE out[N_MAT])
{
    int i, j;
    TYPE sum, Si;

    int id = get_global_id(0); //id = row;
    //Loop is distributed across kernels spmv_1 : for(i = 0; i < N_MAT; i++){
    sum = 0; Si = 0;
    int tmp_begin = rowDelimiters[id];
    int tmp_end = rowDelimiters[id+1];
    for (j = tmp_begin; j < tmp_end; j++){
        Si = val[j] * vec[cols[j]];
        sum = sum + Si;
    }
    out[id] = sum;
    //}
}

/*
    Vector Addition Kernel Implementation using dataflow 
    Arguments:
        in   (input)  --> Input Vector
        out  (output) --> Output Vector
        inc  (input)  --> Increment
        size (input)  --> Size of Vector in Integer
   */
__kernel 
void spmv(__global TYPE val[NNZ], __global int cols[NNZ], __global int rowDelimiters[N_MAT+1], __global TYPE vec[N_MAT], __global TYPE out[N_MAT])
{
    // float buffer_in1[BUFFER_SIZE];
    // float buffer_in2[BUFFER_SIZE];
    // float buffer_out[BUFFER_SIZE];

    // read_input(in1,buffer_in1,size);
    // read_input(in2,buffer_in2,size);
    compute_spmv(val, cols, rowDelimiters, vec, out);
    // write_result(out,buffer_out,size);
}
