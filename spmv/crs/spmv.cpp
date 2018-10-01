/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    //OPENCL HOST CODE AREA START
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 

    //Create Program and Kernel
    std::string binaryFile = xcl::find_binary_file(device_name,"spmv");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl_spmv(program,"spmv");

    //Allocate Buffer in Global Memory
    cl::Buffer buffer_val (context, CL_MEM_READ_ONLY,
                        NNZ * sizeof(TYPE));
    cl::Buffer buffer_cols (context, CL_MEM_READ_ONLY,
                        (NNZ) * sizeof(int32_t));
    cl::Buffer buffer_rowDelimiters (context, CL_MEM_READ_ONLY,
                           (N+1) * sizeof(int32_t));
    cl::Buffer buffer_vec (context, CL_MEM_READ_ONLY,
                           (N) * sizeof(TYPE));
    cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, 
                            N * sizeof(TYPE));

    //Copy input data to device global memory
    q.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, NNZ * sizeof(TYPE), val);
    q.enqueueWriteBuffer(buffer_cols, CL_TRUE, 0, NNZ * sizeof(int32_t), cols);
    q.enqueueWriteBuffer(buffer_rowDelimiters, CL_TRUE, 0,  (N+1) * sizeof(int32_t), rowDelimiters);
    q.enqueueWriteBuffer(buffer_vec, CL_TRUE, 0,  (N) * sizeof(TYPE), vec);

   // int inc = INCR_VALUE;
    int size = DATA_SIZE;
    int cols = COLS;
    //Set the Kernel Arguments
    int narg=0;
    krnl_spmv.setArg(narg++,buffer_val);
    krnl_spmv.setArg(narg++,buffer_cols);
    krnl_spmv.setArg(narg++,buffer_rowDelimiters);
    krnl_spmv.setArg(narg++,buffer_vec);
    krnl_spmv.setArg(narg++,buffer_out);

    //Launch the Kernel
    q.enqueueNDRangeKernel(krnl_spmv,cl::NullRange,cl::NDRange(cols,size/cols),cl::NullRange);

    //Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(out, CL_TRUE, 0, N * sizeof(TYPE), out);

    q.finish();
}


