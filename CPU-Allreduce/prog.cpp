#include <iostream>
#include <vector>
#include "CPrecisionTimer.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <assert.h>

int main(int argc, char** argv) 
{
	std::cout << "Hello" << std::endl;
	//unsigned long long elem = 2000592040;  // 7.45GB
	//unsigned long long elem = 1073741824; // 4GB
	//unsigned long long elem = 536870912; // 2GB
	//unsigned long long elem = 268435456; // 1GB
	unsigned long long elem = 131072000; // 500 MB
	std::vector<float> grad_array(elem, 0.0123456789);
	
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
	// Allocate the device memory
    float *d_grad = NULL;
	err = cudaMalloc((void **)&d_grad, grad_array.size()*sizeof(float));
	assert(err == cudaSuccess);

	MPI::Init(argc, argv);
	int id = MPI::COMM_WORLD.Get_rank();
	int sz = MPI::COMM_WORLD.Get_size();

	std::cout << "CPU Allreduce Start." << std::endl;

	MPI::COMM_WORLD.Barrier();
	CPrecisionTimer Total_T;
	Total_T.Start();
	
	// Copy Identical CPU grad_array to GPU ========================
	err = cudaMemcpy(d_grad, &grad_array[0], grad_array.size()*sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
    
	// Copy Back GPU grad_array to CPU =============================
	err = cudaMemcpy(&grad_array[0], d_grad, grad_array.size()*sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);

	// Allreduce on CPU ============================================
	MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &grad_array[0], grad_array.size(), MPI::FLOAT, MPI::SUM);
	
	// Copy CPU result to GPU ==========================================
	err = cudaMemcpy(d_grad, &grad_array[0], grad_array.size()*sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);

	// Copy Back GPU grad_array to CPU =============================
	err = cudaMemcpy(&grad_array[0], d_grad, grad_array.size()*sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);


	MPI::COMM_WORLD.Barrier();
	std::cout << "Total CPU Allreduce Time = " << Total_T.Stop() << " seconds. " << std::endl;

	MPI::Finalize();

	return 0;
}

