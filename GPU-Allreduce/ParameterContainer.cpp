//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <iostream>
#include <assert.h>
#include "ParameterContainer.h"
#include "../CPrecisionTimer.h"
#include "LSDN_mathfunctions.h"

// added by Youjie Li
#include "../MY_COMMON/myflags.h"
#include <fstream>

#ifdef WITH_GPU
#include "cuda_runtime.h"
#include "../LSDN_CudaCommon.h"
#define DEF_CUDA_FREE(x) \
if ((x) != NULL) { \
	cudaFree((x)); \
	(x) = NULL; \
}
#else
#define DEF_CUDA_FREE(x) \
	assert((x)==NULL);
#endif

#ifdef WITH_MPI
#include "mpi.h"
#endif

template <typename V, typename S, bool G, bool P>
ParameterContainer<V, S, G, P>::ParameterContainer() : GPUParameterValues(NULL), GPUParameterDerivative(NULL), ParameterDiffHistory(NULL), ParameterWorkingSetData(NULL), GPUParameterValuesRootOffset(0) {}

template <typename V, typename S, bool G, bool P>
ParameterContainer<V, S, G, P>::~ParameterContainer() {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::Clear() {
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		(*it)->DeletedByContainer();
		(*it)->Clear();
		delete *it;
	}

	DEF_CUDA_FREE(GPUParameterValues)
	DEF_CUDA_FREE(GPUParameterDerivative)

	if (ParameterDiffHistory != NULL) {
		DeAllocValueMem(typename NodeType::GPUType(), ParameterDiffHistory);
		ParameterDiffHistory = NULL;
	}

	if (ParameterWorkingSetData != NULL) {
		DeAllocValueMem(typename NodeType::GPUType(), ParameterWorkingSetData);
		ParameterWorkingSetData = NULL;
	}
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::DeAllocValueMem(i2t<true>, ValueType* ptr) {
	cudaFree(ptr);
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::DeAllocValueMem(i2t<false>, ValueType* ptr) {
	delete[] ptr;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ReduceStepSize() {
	for (typename std::vector<ParaType*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		(*it)->ReduceStepSize();
	}
}

template <typename V, typename S, bool G, bool P>
typename ParameterContainer<V, S, G, P>::ParaType* ParameterContainer<V, S, G, P>::AddParameter(SizeType* sz, SizeType numDim, const typename ParaType::NodeParameters& params, bool isRoot, int paramID) {
	ParaType* retVal = new ParaType(params);
	retVal->SetValueSize(NULL, sz, numDim, 0, 0);
	ParameterIDToContainerPosition.insert(std::pair<int, int>(paramID, int(paramClasses.size())));
	paramClasses.push_back(retVal);
	ParamClassesRoot.push_back(isRoot);
	return retVal;
}

template <typename V, typename S, bool G, bool P>
int ParameterContainer<V, S, G, P>::CreateCPUMemoryForParameters() {
	size_t numData = 0;
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		numData += (*it)->GetNumEl(0, 0);
	}
	CPUParameterValues.resize(numData);
	this->UtilityClass<V, S, G, P>::UpdateCPUDataOffset(&CPUParameterValues[0], paramClasses, ParamClassesRoot, false, 0);
	return 0;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::PrepareComputation(STATE purpose) {
	ValueType* baseParameterPtr = &CPUParameterValues[0];
	CreateGPUParameterData(typename NodeType::GPUType(), baseParameterPtr);
	this->UtilityClass<V, S, G, P>::AdjustComputationPointers(baseParameterPtr, paramClasses, ParamClassesRoot, &GPUParameterValuesRootOffset, 0);

	if (purpose == TRAIN) {
		CreateCPUDerivativeMemoryForParameters(typename NodeType::GPUType());
		ValueType* baseParameterDerivativePtr = NULL;
		ValueType* ParameterRootOffset = NULL;
		if (CPUParameterDerivative.size() > 0) {//mod-ify: no modification required
			baseParameterDerivativePtr = &CPUParameterDerivative[0];
			ParameterRootOffset = baseParameterDerivativePtr + ComputeRootFunctionOffset(typename NodeType::GPUType(), GPUParameterValuesRootOffset);
		}
		CreateGPUDerivativeMemoryForParameters(typename NodeType::GPUType(), baseParameterDerivativePtr);
		this->UtilityClass<V, S, G, P>::AdjustDerivativePointers(baseParameterDerivativePtr, paramClasses, ParamClassesRoot, ParameterRootOffset, 0);//mod-ify: make sure root pointers fit in case of MPI; adjusted ComputeRootFunctionOffset(i2t<true>, size_t val)

		CreateHistoryData(typename NodeType::GPUType());
		AdjustHistoryPointers();

		SizeType numData = 0;
		for (typename std::vector<ParaType*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
			numData = std::max((*it)->WorkingSetRequirement(), numData);
		}
		if (numData > 0) {
			CreateWorkingSetData(typename NodeType::GPUType(), numData);
		}
	}
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateWorkingSetData(i2t<true>, SizeType numData) {
	cudaMalloc(&ParameterWorkingSetData, numData*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	cudaMemset(ParameterWorkingSetData, 0, numData*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateWorkingSetData(i2t<false>, SizeType numData) {
	ParameterWorkingSetData = new ValueType[numData];
	std::fill(ParameterWorkingSetData, ParameterWorkingSetData + numData, ValueType(0.0));
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
#ifdef WITH_MPI
size_t ParameterContainer<V, S, G, P>::ComputeRootFunctionOffset(i2t<true>, size_t val) {
	return val;
#else
size_t ParameterContainer<V, S, G, P>::ComputeRootFunctionOffset(i2t<true>, size_t) {
	return 0;
#endif
}
#endif

template <typename V, typename S, bool G, bool P>
size_t ParameterContainer<V, S, G, P>::ComputeRootFunctionOffset(i2t<false>, size_t val) {
	return val;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::AdjustHistoryPointers() {
	size_t numData = 0;
	std::vector<bool>::iterator isRoot = ParamClassesRoot.begin();
	for (typename std::vector<ParaType*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it, ++isRoot) {
		if (!*isRoot) {
			ValueType** output = (*it)->GetDiffHist();
			assert(output != NULL && *output == NULL);
			*output = ParameterDiffHistory + numData;
			numData += (*it)->GetNumEl(0, 0);
		}
	}
	isRoot = ParamClassesRoot.begin();
	for (typename std::vector<ParaType*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it, ++isRoot) {
		if (*isRoot) {
			ValueType** output = (*it)->GetDiffHist();
			assert(output != NULL && *output == NULL);
			*output = ParameterDiffHistory + numData;
			numData += (*it)->GetNumEl(0, 0);
		}
	}
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUParameterData(i2t<true>, ValueType*& baseDataPtr) {
	assert(GPUParameterValues == NULL);
	cudaMalloc(&GPUParameterValues, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	cudaMemcpy(GPUParameterValues, &CPUParameterValues[0], CPUParameterValues.size()*sizeof(ValueType), cudaMemcpyHostToDevice);
	check_cuda_errors(__FILE__, __LINE__);
	baseDataPtr = GPUParameterValues;
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUParameterData(i2t<false>, ValueType*&) {}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateHistoryData(i2t<true>) {
	assert(ParameterDiffHistory == NULL);
	cudaMalloc(&ParameterDiffHistory, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	cudaMemset(ParameterDiffHistory, 0, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateHistoryData(i2t<false>) {
	assert(ParameterDiffHistory == NULL);
	ParameterDiffHistory = new ValueType[CPUParameterValues.size()];
	std::fill(ParameterDiffHistory, ParameterDiffHistory + CPUParameterValues.size(), ValueType(0.0));
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUDerivativeMemoryForParameters(i2t<true>, ValueType*& baseDataPtr) {
	assert(GPUParameterDerivative == NULL);
	cudaMalloc(&GPUParameterDerivative, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	cudaMemset(GPUParameterDerivative, 0, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	baseDataPtr = GPUParameterDerivative;
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUDerivativeMemoryForParameters(i2t<false>, ValueType*&) {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateCPUDerivativeMemoryForParameters(i2t<false>) {
	CPUParameterDerivative.resize(CPUParameterValues.size(), ValueType(0.0));
	//std::cout<< "CPUParameterDerivative.size() = " << CPUParameterDerivative.size() << std::endl;
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateCPUDerivativeMemoryForParameters(i2t<true>) {//mod-ify: generate complete array in case of MPI (not only root values)
	size_t numEl = 0;
	std::vector<bool>::iterator isRoot = ParamClassesRoot.begin();
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it, ++isRoot) {
#ifdef WITH_MPI
		numEl += (*it)->GetNumEl(0, 0);
#else
		if (*isRoot) {
			numEl += (*it)->GetNumEl(0, 0);
		}
#endif
	}
	CPUParameterDerivative.resize(numEl, ValueType(0.0));
}
#endif

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyRootParameterValues(i2t<true>) {
	if (CPUParameterValues.size()>GPUParameterValuesRootOffset) {
		cudaMemcpy(&CPUParameterValues[GPUParameterValuesRootOffset], GPUParameterValues + GPUParameterValuesRootOffset, (CPUParameterValues.size() - GPUParameterValuesRootOffset)*sizeof(ValueType), cudaMemcpyDeviceToHost);
		check_cuda_errors(__FILE__, __LINE__);
	}
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyRootParameterValues(i2t<false>) {}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>//mod-ify: copy to the right location in case of MPI
void ParameterContainer<V, S, G, P>::CopyRootParameterDerivatives(i2t<true>) {//assumes size of CPUParameterDerivative to equal number of root elements
#ifdef WITH_MPI
	if (GPUParameterValuesRootOffset < CPUParameterValues.size()) {
		cudaMemcpy(GPUParameterDerivative + GPUParameterValuesRootOffset, &CPUParameterDerivative[0] + GPUParameterValuesRootOffset, (CPUParameterDerivative.size() - GPUParameterValuesRootOffset)*sizeof(ValueType), cudaMemcpyHostToDevice);
		check_cuda_errors(__FILE__, __LINE__);
	}
#else
	if (CPUParameterDerivative.size() > 0) {
		cudaMemcpy(GPUParameterDerivative + GPUParameterValuesRootOffset, &CPUParameterDerivative[0], CPUParameterDerivative.size()*sizeof(ValueType), cudaMemcpyHostToDevice);
		check_cuda_errors(__FILE__, __LINE__);
	}
#endif
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyRootParameterDerivatives(i2t<false>) {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesFromGPU(i2t<false>) {}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesFromGPU(i2t<true>) {
#ifdef WITH_MPI
	cudaMemcpy(&CPUParameterDerivative[0], GPUParameterDerivative, CPUParameterDerivative.size()*sizeof(ValueType), cudaMemcpyDeviceToHost);
	check_cuda_errors(__FILE__, __LINE__);
#else
	assert(false);
#endif
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesToGPU(i2t<false>) {}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesToGPU(i2t<true>) {
#ifdef WITH_MPI
	cudaMemcpyAsync(GPUParameterDerivative, &CPUParameterDerivative[0], CPUParameterDerivative.size()*sizeof(ValueType), cudaMemcpyHostToDevice);
	check_cuda_errors(__FILE__, __LINE__);
#else
	assert(false);
#endif
}
#endif

#ifdef WITH_MPI
#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::PerformCopy(i2t<true>) {
	//cudaDeviceSynchronize();
	cudaStreamSynchronize((cudaStream_t)0);//computation happens on stream 0; luckily not a special stream anymore since CUDA 7.0
	//compatible GPU and MPI implementation required; careful checks necessary;
	if (sizeof(ValueType) == 4) // the same as below
	{
		//MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, GPUParameterDerivative, CPUParameterDerivative.size(), MPI::FLOAT, MPI::SUM);
		void* vptr = &CPUParameterDerivative;
		std::vector<float>* Gradients_ptr = (std::vector<float>*) vptr;
		FloatPerformCopy(*Gradients_ptr, CPUParameterDerivative.size());
	} 
	else if (sizeof(ValueType) == 8) 
	{
		assert(false); // for safety
		MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, GPUParameterDerivative, CPUParameterDerivative.size(), MPI::DOUBLE, MPI::SUM);
	} 
	else 
	{
		assert(false);
	}
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::PerformCopy(i2t<false>) 
{
	if (sizeof(ValueType) == 4) 
	{
		// added by Youjie
		void* vptr = &CPUParameterDerivative;
		std::vector<float>* Gradients_ptr = (std::vector<float>*) vptr;
		FloatPerformCopy(*Gradients_ptr, CPUParameterDerivative.size());
	} 
	else if (sizeof(ValueType) == 8) 
	{
		assert(false); // for safety
		MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &CPUParameterDerivative[0], CPUParameterDerivative.size(), MPI::DOUBLE, MPI::SUM);
	} 
	else 
	{
		assert(false);
	}
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::Update(int ClusterID, int ClusterSize)
{//mod-ify: copy derivative from GPU to CPU; MPIAllreduce; copy back to GPU
	if (ClusterSize > 1) 
	{
#ifdef WITH_MPI
#ifdef WITH_DIRECT_GPU_COPY
		//compatible GPU and MPI implementation required; careful checks necessary;
		PerformCopy(typename NodeType::GPUType());
#else
		#ifdef GET_BREAKDOWN //==========================================
		CPrecisionTimer CGTimer; CGTimer.Start();
		#endif
		CopyParameterDerivativesFromGPU(typename NodeType::GPUType()); // Copy Gradients GPU/CPU -> CPU
		#ifdef GET_BREAKDOWN
    	total_CPUGPUtrans += CGTimer.Stop();
    	#endif //========================================================

		#ifdef GET_BREAKDOWN //==============================
		CPrecisionTimer perfcopyTimer; perfcopyTimer.Start();
		#endif
		PerformCopy(i2t<false>());
		#ifdef GET_BREAKDOWN
    	total_performcopy += perfcopyTimer.Stop();
    	#endif //============================================

		#ifdef GET_BREAKDOWN //==========================================
		CGTimer.Start();
		#endif
		CopyParameterDerivativesToGPU(typename NodeType::GPUType()); // Copy Gradients CPU -> GPU/CPU
		#ifdef GET_BREAKDOWN
    	total_CPUGPUtrans += CGTimer.Stop();
    	#endif //========================================================
#endif
#endif
	}

	//Debug
	/*if (ClusterID==ClusterSize-1)
	{
		GetWeights(typename NodeType::GPUType()); // Get Weight from CPU/GPU to CPUParameterValues 
		std::ofstream ofs("ExactPS_init_model", std::ios_base::binary | std::ios_base::out);
		if(!ofs.is_open()) 
		{
			std::cout << "Error opening output file: " << std::endl;
		} 
		else 
		{
			ofs.write((char*)&CPUParameterValues[0], CPUParameterValues.size()*sizeof(ValueType));
			ofs.close();
		}	
	}*/

	// Update Parameters on GPU/CPU
	#ifdef GET_BREAKDOWN //==========================================
	CPrecisionTimer UpdateParaTimer; UpdateParaTimer.Start();
	#endif
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) 
	{
		(*it)->UpdateParameters(ParameterWorkingSetData); //GPU/CPU
	}
	#ifdef GET_BREAKDOWN
    total_updateparas += UpdateParaTimer.Stop();
    #endif //========================================================

#ifdef EXACT_PS
	int PS = ClusterSize-1;
	assert(sizeof(ValueType)==4); // must be float type
	assert(ClusterSize > 1); //Must be parallel code
	
    // 1. Transfer PS's Updated_Parameter to CPU
	if ( ClusterID == PS )
		GetWeights(typename NodeType::GPUType()); // Get Weight from CPU/GPU to CPUParameterValues 
	 
	#ifdef TRUNCT_WEIGHT
		#if defined TENBIT_TRUNCAT
			std::vector<float> comp(CPUParameterValues.size()/3+1, 0);
			void* vptr  = (void*) &CPUParameterValues[0];
			float* flt_src = (float*) vptr;
			// 2. Truncate PS's Updated_Paramters
			if ( ClusterID == PS )
			{
				tenbit_truncat_without( flt_src, CPUParameterValues.size(), &comp[0], comp.size());
			}
			// 3. PS Bcast to all
			MPI::COMM_WORLD.Bcast(&comp[0], comp.size(), MPI::FLOAT, PS);
			// 4. All Processes Restore Truncated Paramters
			restore_truncat_without(&comp[0], comp.size(), flt_src, CPUParameterValues.size());
		#elif defined QUAD_TRUNCAT
			std::vector<unsigned char> comp(CPUParameterValues.size(), 0);
			void* vptr  = (void*) &CPUParameterValues[0];
			float* flt_src = (float*) vptr;
			// 2. Truncate PS's Updated_Paramters
			if ( ClusterID == PS )
			{
				quad_truncat(flt_src, CPUParameterValues.size(), &comp[0]);
			}
			// 3. Root Bcast to all
			MPI::COMM_WORLD.Bcast(&comp[0], comp.size(), MPI::UNSIGNED_CHAR, PS);
			// 4. All Processes Restore Truncated Paramters
			restore_quad_truncat(&comp[0], comp.size(), flt_src);
		#elif defined HALF_TRUNCAT
			std::vector<unsigned short> comp(CPUParameterValues.size(), 0);
			void* vptr  = (void*) &CPUParameterValues[0];
			float* flt_src = (float*) vptr;
			// 2. Truncate PS's Updated_Paramters
			if ( ClusterID == PS )
			{
				half_truncat(flt_src, CPUParameterValues.size(), &comp[0]);
			}
			// 3. PS Bcast to all
			MPI::COMM_WORLD.Bcast(&comp[0], comp.size(), MPI::UNSIGNED_SHORT, PS);
			// 4. All Processes Restore Truncated Paramters
			restore_half_truncat(&comp[0], comp.size(), flt_src);		
		#endif
	#else //(default): full-precision on Updated parameters*/
		// 3. PS Bcast to all
		MPI::COMM_WORLD.Bcast(&CPUParameterValues[0], CPUParameterValues.size(), MPI::FLOAT, PS);
	#endif
	// 5. All Processes Overwrite Old Parameters on CPU/GPU
	// i2t<flase>: Set this weight to CPUParameterValues
	// i2t<true>: Set this weight to GPUParameterValues
	SetWeights(typename NodeType::GPUType(), &CPUParameterValues); 

#endif

}

template <typename V, typename S, bool G, bool P>
V ParameterContainer<V, S, G, P>::GetRegularization() {
	ValueType reg = 0;
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		reg += (*it)->GetRegularization();
	}
	return reg;
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetGradient(i2t<true>) {
	if (GPUParameterDerivative != NULL) {
		cudaMemset(GPUParameterDerivative, 0, sizeof(ValueType)*GPUParameterValuesRootOffset);//not required to clear root part on GPU since it's computed on the CPU and then copied to GPU; CPU memory is cleared in ResetCPURootParameterDerivative(i2t<true>)
		//cudaMemset(GPUParameterDerivative, 0, sizeof(ValueType)*CPUParameterValues.size());
		check_cuda_errors(__FILE__, __LINE__);
	}
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetGradient(i2t<false>) {
	std::fill(CPUParameterDerivative.begin(), CPUParameterDerivative.begin() + GPUParameterValuesRootOffset, ValueType(0.0));//remaining part will be cleared by ResetCPURootParameterDerivative(i2t<false>)
	//std::fill(CPUParameterDerivative.begin(), CPUParameterDerivative.end(), ValueType(0.0));
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetCPURootParameterDerivative(i2t<true>) {//mod-ify: in case of MPI we only clear the root part and not everything
#ifdef WITH_MPI
	std::fill(CPUParameterDerivative.begin() + GPUParameterValuesRootOffset, CPUParameterDerivative.end(), ValueType(0.0));
#else
	std::fill(CPUParameterDerivative.begin(), CPUParameterDerivative.end(), ValueType(0.0));
#endif
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetCPURootParameterDerivative(i2t<false>) {
	std::fill(CPUParameterDerivative.begin() + GPUParameterValuesRootOffset, CPUParameterDerivative.end(), ValueType(0.0));
}

template <typename V, typename S, bool G, bool P>
typename std::vector<typename ParameterContainer<V, S, G, P>::ParaType*>* ParameterContainer<V, S, G, P>::GetParamClasses() {
	return &paramClasses;
}

template <typename V, typename S, bool G, bool P>
typename std::vector<typename ParameterContainer<V, S, G, P>::ValueType>* ParameterContainer<V, S, G, P>::GetWeights(i2t<false>) {
	return &CPUParameterValues;
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
typename std::vector<typename ParameterContainer<V, S, G, P>::ValueType>* ParameterContainer<V, S, G, P>::GetWeights(i2t<true>) {
	cudaMemcpy(&CPUParameterValues[0], GPUParameterValues, CPUParameterValues.size()*sizeof(ValueType), cudaMemcpyDeviceToHost);
	check_cuda_errors(__FILE__, __LINE__);
	return &CPUParameterValues;
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::SetWeights(i2t<false>, std::vector<ValueType>* weights) {
	assert(weights->size() == CPUParameterValues.size());
	std::copy(weights->begin(), weights->end(), CPUParameterValues.begin());
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::SetWeights(i2t<true>, std::vector<ValueType>* weights) {
	assert(weights->size() == CPUParameterValues.size());
	cudaMemcpy(GPUParameterValues, &((*weights)[0]), weights->size()*sizeof(ValueType), cudaMemcpyHostToDevice);
	check_cuda_errors(__FILE__, __LINE__);
}
#endif

template <typename V, typename S, bool G, bool P>
size_t ParameterContainer<V, S, G, P>::GetWeightDimension() {
	return CPUParameterValues.size();
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::GetDerivative(i2t<false>, std::vector<ValueType>& deriv) {
	deriv.assign(CPUParameterDerivative.begin(), CPUParameterDerivative.end());
}

#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::GetDerivative(i2t<true>, std::vector<ValueType>& deriv) {
	deriv.assign(CPUParameterValues.size(), ValueType(0.0));
	cudaMemcpy(&deriv[0], GPUParameterDerivative, deriv.size()*sizeof(ValueType), cudaMemcpyDeviceToHost);
	check_cuda_errors(__FILE__, __LINE__);
}
#endif

template <typename V, typename S, bool G, bool P>
typename ParameterContainer<V, S, G, P>::ParaType* ParameterContainer<V, S, G, P>::GetPtrFromID(int id) {
	std::map<int, int>::iterator iter = ParameterIDToContainerPosition.find(id);
	if (iter == ParameterIDToContainerPosition.end()) {
		return NULL;
	} else {
		return paramClasses[iter->second];
	}
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::MultiplyStepSize(ValueType smul) {
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		(*it)->MultiplyStepSize(smul);
	}
}

// added by Renjie
#ifdef WITH_GPU
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetHistoryData(i2t<true>) {
	assert(ParameterDiffHistory != NULL);
	cudaMemset(ParameterDiffHistory, 0, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
}
#endif

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetHistoryData(i2t<false>) {
	assert(ParameterDiffHistory != NULL);	
	std::fill(ParameterDiffHistory, ParameterDiffHistory + CPUParameterValues.size(), ValueType(0.0));
}

//++++++++++++++ added by Youjie +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#if defined EXACT_PS //////// Exact Parameter Server Row //////////////// /* Memory Allocation to be Optimized, No Timing Breakdown yet*/ 
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::FloatPerformCopy(std::vector<float> & Gradients, size_t GradSize)
{
	// Gather Gradients from all workers to P.S
	int ClusterSize = MPI::COMM_WORLD.Get_size();
	int ClusterID = MPI::COMM_WORLD.Get_rank();
	int PS = ClusterSize-1;
	
	#ifdef TRUNCT_GRAD
		#if defined TENBIT_TRUNCAT
			// 1. All Processes Truncate Gradients
			std::vector<float> comp(GradSize/3+1, 0.0);
			tenbit_truncat_without( &Gradients[0], GradSize, &comp[0], comp.size());
			// 2. Gather Truncated Gradients to PS
			//if (ClusterID == PS)
			std::vector<float> recv_buffer(ClusterSize*comp.size(), 0); 
			MPI::COMM_WORLD.Gather(&comp[0], comp.size(), MPI::FLOAT, 
								&recv_buffer[0], comp.size(), MPI::FLOAT, PS);
			// 3. PS Restores Truncated Gradients
			std::vector<float> full_buffer(ClusterSize*GradSize, 0.0);
			if (ClusterID == PS)
			{
				for (int rid = 0; rid < ClusterSize; rid ++)
					restore_truncat_without(&recv_buffer[rid*comp.size()], comp.size(), &full_buffer[rid*GradSize], GradSize);
			}
		#elif defined QUAD_TRUNCAT
			// 1. All Processes Truncate Gradients
			std::vector<unsigned char> comp(GradSize, 0);
			quad_truncat(&Gradients[0], GradSize, &comp[0]);
			// 2. Gather Truncated Gradients to PS
			//if (ClusterID == PS)
			std::vector<unsigned char> recv_buffer(ClusterSize*comp.size(), 0); 
			MPI::COMM_WORLD.Gather(&comp[0], comp.size(), MPI::UNSIGNED_CHAR, 
								&recv_buffer[0], comp.size(), MPI::UNSIGNED_CHAR, PS);
			// 3. PS Restores Truncated Gradients
			std::vector<float> full_buffer(ClusterSize*GradSize, 0.0);
			if (ClusterID == PS)
			{
				restore_quad_truncat(&recv_buffer[0], recv_buffer.size(), &full_buffer[0]);
			}
		#elif defined HALF_TRUNCAT
			// 1. All Processes Truncate Gradients
			std::vector<unsigned short> comp(GradSize, 0);
			half_truncat(&Gradients[0], GradSize, &comp[0]);
			// 2. Gather Truncated Gradients to PS
			//if (ClusterID == PS)
			std::vector<unsigned short> recv_buffer(ClusterSize*comp.size(), 0); 
			MPI::COMM_WORLD.Gather(&comp[0], comp.size(), MPI::UNSIGNED_SHORT, 
								&recv_buffer[0], comp.size(), MPI::UNSIGNED_SHORT, PS);
			// 3. PS Restores Truncated Gradients
			std::vector<float> full_buffer(ClusterSize*GradSize, 0.0);
			if (ClusterID == PS)
			{
				restore_half_truncat(&recv_buffer[0], recv_buffer.size(), &full_buffer[0]);
			}				
		#endif
	#else //(default): full-precision on Gradient */
		std::vector<float> full_buffer(ClusterSize*GradSize, 0.0);
		//Gather full precision gradient arrays to P.S
		MPI::COMM_WORLD.Gather(&Gradients[0], GradSize, MPI::FLOAT, 
							 &full_buffer[0], GradSize, MPI::FLOAT, PS);
	#endif

	if (ClusterID==PS)
	{
		// sum up full-precision gradient arrays
		// ResetGradient(i2t<false>()); // reset CPU Gradient
		std::vector<float> II(ClusterSize-1,1.0);
		int dim1 = GradSize, dim2 = 1, dim3 = ClusterSize-1; 
		MultiplyMatMat(i2t<false>(),
					   &full_buffer[0], &II[0], &Gradients[0], dim1, dim2, dim3, 
					   CblasNoTrans, CblasNoTrans, 1.0, 0.0);
		// PS: Total Gradients stored at vector "Gradients" ("CPUParameterDerivatives")
		// Debug write Total Gradient to file
		/*std::ofstream ofs("TotalGrad_ExactPS.bin", std::ios_base::binary | std::ios_base::out);
		if(!ofs.is_open()) 
		{
			std::cout << "Error opening output file: " << std::endl;
			assert(false);
		} 
		else 
		{
			ofs.write((char*)&Gradients[0], GradSize*sizeof(float));
			ofs.close();
		}*/	
		// Workers: Gradients at CPU remains the same
	}

}

#elif defined USE_GATHERBCAST //////// GatherBcast Row /////////////////////////////////////
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::FloatPerformCopy(std::vector<float> & Gradients, size_t GradSize)
{
	// Original MPI (Gather & Bcast)
	int ClusterSize = MPI::COMM_WORLD.Get_size();
	int ClusterID = MPI::COMM_WORLD.Get_rank();
	int Root=0;
	
	std::vector<float> recvbuf(ClusterSize*GradSize, 0.0);
	//Gather full precision gradient arrays to root
	MPI::COMM_WORLD.Gather(&Gradients[0], GradSize, MPI::FLOAT, 
						   &recvbuf[0], GradSize, MPI::FLOAT, Root);
	
	if (ClusterID==Root)
	{
		#ifdef GET_BREAKDOWN //==================
    	CPrecisionTimer accTimer; accTimer.Start();
		#endif 
		// sum up gradient arrays
		// ResetGradient(i2t<false>()); // reset CPU Gradient
		std::vector<float> II(ClusterSize,1.0);
		int dim1 = GradSize, dim2 = 1, dim3 = ClusterSize; 
		MultiplyMatMat(i2t<false>(), /*CPU version*/
					   &recvbuf[0], &II[0], &Gradients[0], dim1, dim2, dim3, 
					   CblasNoTrans, CblasNoTrans, 1.0, 0.0);
		#ifdef GET_BREAKDOWN
    	total_accum += accTimer.Stop();
    	#endif //=================================
	}

	//Root Bcast Summation to all other nodes
	MPI::COMM_WORLD.Bcast(&Gradients[0], GradSize, MPI::FLOAT, Root);
}

#elif defined USE_ALLGATHER //////// Allgather Row ///////////////////////////////////////
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::FloatPerformCopy(std::vector<float> & Gradients, size_t GradSize)
{
	// Original MPI (Allgather)
	#ifdef GET_BREAKDOWN //==================================
	CPrecisionTimer allgaTimer; allgaTimer.Start();
	#endif 

	int np = MPI::COMM_WORLD.Get_size();
	std::vector<float> recvbuf(np*GradSize, 0.0);
	MPI::COMM_WORLD.Allgather(&Gradients[0], GradSize, MPI::FLOAT, 
							  &recvbuf[0], GradSize, MPI::FLOAT);
	#ifdef GET_BREAKDOWN
	total_allgather += allgaTimer.Stop();
	#endif //================================================

	#ifdef GET_BREAKDOWN //==================
    CPrecisionTimer accTimer; accTimer.Start();
	#endif 
	// sum up gradient arrays
	ResetGradient(i2t<false>()); // reset CPU Gradient
	for (unsigned int i=0; i < np; ++i)
	{
		VectorAdd(i2t<false>(), GradSize, 1.0, &recvbuf[i*GradSize], &Gradients[0]); // CPU version
	}
	#ifdef GET_BREAKDOWN
    total_accum += accTimer.Stop();
    #endif //=================================

}

#else //////// Allreduce Row /////////////////////////////////////////////////////////////
template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::FloatPerformCopy(std::vector<float> & Gradients, size_t GradSize)
{
#if defined USE_MYSUM

  #if defined TENBIT_TRUNCAT // 10/32 sized array 

  	#ifdef GET_BREAKDOWN //==================================
	CPrecisionTimer truncatTimer; truncatTimer.Start();
	#endif   
	std::vector<float> comp(Gradients.size()/3+1, 0.0);
	tenbit_truncat_without( &Gradients[0],  Gradients.size(), &comp[0], comp.size());

	#ifdef GET_BREAKDOWN //==================		
	CPrecisionTimer allredTimer; allredTimer.Start();
	#endif
	// Allreduce 1/3-sized array
	MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &comp[0], comp.size(), MPI::FLOAT, myOp);
	//
	#ifdef GET_BREAKDOWN
	double cur_allredTime = allredTimer.Stop();
	total_allreduce += cur_allredTime;
	#endif //================================
	
	// restore 10bit-truncation
	restore_truncat_without(&comp[0], comp.size(), &Gradients[0], Gradients.size());
	#ifdef GET_BREAKDOWN
	total_truncat_without += truncatTimer.Stop() - cur_allredTime;
	#endif //===================================================
  
  #elif defined QUAD_TRUNCAT // quad-sized array

  	#ifdef GET_BREAKDOWN //==================================
	CPrecisionTimer truncatTimer; truncatTimer.Start();
	#endif   
	// quad-truncation
	std::vector<unsigned char> comp(GradSize, 0);
	quad_truncat(&Gradients[0], GradSize, &comp[0]);
	
	#ifdef GET_BREAKDOWN //==================		
	CPrecisionTimer allredTimer; allredTimer.Start();
	#endif
	// Allreduce quad-sized array
	MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &comp[0], comp.size(), MPI::UNSIGNED_CHAR, myOp);
	//
	#ifdef GET_BREAKDOWN
	double cur_allredTime = allredTimer.Stop();
	total_allreduce += cur_allredTime;
	#endif //================================

	// restore quad-truncation
	restore_quad_truncat(&comp[0], comp.size(), &Gradients[0]);
	#ifdef GET_BREAKDOWN
	total_truncat_without += truncatTimer.Stop() - cur_allredTime;
	#endif //===================================================
	
  #elif defined HALF_TRUNCAT	// half-sized array

	#ifdef GET_BREAKDOWN //==================================
	CPrecisionTimer truncatTimer; truncatTimer.Start();
	#endif   
	// half-truncation
	std::vector<unsigned short> comp(GradSize, 0);
	half_truncat(&Gradients[0], GradSize, &comp[0]);
	
	#ifdef GET_BREAKDOWN //==================		
	CPrecisionTimer allredTimer; allredTimer.Start();
	#endif
	// Allreduce half-sized array
	MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &comp[0], comp.size(), MPI::UNSIGNED_SHORT, myOp);
	//
	#ifdef GET_BREAKDOWN
	double cur_allredTime = allredTimer.Stop();
	total_allreduce += cur_allredTime;
	#endif //================================

	// restore half-truncation
	restore_half_truncat(&comp[0], comp.size(), &Gradients[0]);
	#ifdef GET_BREAKDOWN
	total_truncat_without += truncatTimer.Stop() - cur_allredTime;
	#endif //===================================================

  #else // CustomSum with full-sized array

	#ifdef GET_BREAKDOWN //==================		
	CPrecisionTimer myTimer; myTimer.Start();
	#endif 

	MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &Gradients[0], GradSize, MPI::FLOAT, myOp);
	
	#ifdef GET_BREAKDOWN
	total_allreduce += myTimer.Stop();
	#endif //================================

  #endif

#elif defined USE_COMPRESSION /* No Timing Breakdown yet*/

    int ClusterID = MPI::COMM_WORLD.Get_rank();
	
	#ifdef GET_STATIS
	if (ClusterID==0)
	{	
		save_gradient(Gradients, statis_path_root); // save root gradient to this path
		save_usigma(Gradients, statis_path_usigma); // save mean and standard deviation of this gradient
	}
	#endif

	// Original MPI (Allreduce)
	MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &Gradients[0], GradSize, MPI::FLOAT, MPI::SUM);

	#ifdef GET_STATIS
	if (ClusterID==0)
	{	
		save_gradient(Gradients, statis_path_total); // save total gradient to this path
	}
	#endif

    // SZ-1.4.10 Compression
	int r5=0,r4=0,r3=0,r2=0,r1=GradSize;
    size_t outSize = -1;
    unsigned char* compressed_bytes = SZ_compress(SZ_FLOAT, &Gradients[0], &outSize, r5, r4, r3, r2, r1);
	// Root display Compression Ratio
    if (ClusterID==0)
		std::cout << "Compression Ratio = " << GradSize*4.0/outSize << std::endl;
	// SZ-1.4.10 Decompression
	SZ_decompress_args(SZ_FLOAT, compressed_bytes, outSize, &Gradients[0], r5, r4, r3, r2, r1);
	free(compressed_bytes);
	
	#ifdef GET_STATIS
	if (ClusterID==0)
	{	
		save_gradient(Gradients, statis_path_decomp); // save decompressed gradient to this path
	}
	#endif

#else // Original MPI (Allreduce)

	int ClusterID = MPI::COMM_WORLD.Get_rank();
	
	#ifdef GET_STATIS
	if (ClusterID==0)
	{	
		//save_gradient(Gradients, statis_path_root); // save root gradient to this path
		save_usigma(Gradients, statis_path_usigma); // save mean and standard deviation of this gradient
	}
	#endif

	#ifdef GET_BREAKDOWN //==================		
	CPrecisionTimer myTimer; myTimer.Start();
	#endif

	MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &Gradients[0], GradSize, MPI::FLOAT, MPI::SUM);	
	
	#ifdef GET_BREAKDOWN 
	total_allreduce += myTimer.Stop();
	#endif //================================

#endif		
}

#endif /////////////////////////////////////////////////////////////////////////////////////


template class ParameterContainer<double, int, false, false>;
template class ParameterContainer<float, int, false, false>;
#ifdef WITH_GPU
template class ParameterContainer<double, int, true, false>;
template class ParameterContainer<float, int, true, false>;
#endif