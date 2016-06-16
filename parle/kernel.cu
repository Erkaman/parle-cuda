



#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"
#include "helper_cuda.h"

#include "cudpp.h"

#include "hemi/grid_stride_range.h"
#include "hemi/launch.h"

#include "chag/pp/prefix.cuh"
#include "chag/pp/reduce.cuh"

namespace pp = chag::pp;


const int NUM_TESTS = 10;
const int Tests[NUM_TESTS] = {
	10000,
	50000, 
	100000, 
	200000, 
	500000, 
	1000000,
	2000000,
	5000000,
	10000000,
	20000000,

};

const int TRIALS = 100;


CUDPPHandle cudpp;



void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns,
	bool useChag
	);

int parleHost(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut, bool useChag);

int parleCpu(int *in, int n,
	int* symbolsOut,
	int* countsOut);

__global__ void sumKernel(int *g_countsOut, int n, int* plus, int* minus) {
	for (int i : hemi::grid_stride_range(0, n)) {
		g_countsOut[i] = plus[i] + minus[i];
	}

}

__global__ void scatterKernel(
	int *g_backwardMask, int* g_scannedBackwardMask,
	int *g_forwardMask, int* g_scannedForwardMask,
	int *g_in,
	int *g_symbolsOut, int *g_countsOut, int n, int* plus, int* minus) {

	for (int i : hemi::grid_stride_range(0, n)) {

		int offset;

		if (g_backwardMask[i] == 1){

			offset = g_scannedBackwardMask[i];

			int symbol = g_in[i];

			g_symbolsOut[offset] = symbol;
			//g_countsOut[offset] += -i;
			minus[offset] = -i;
			//atomicAdd(&g_countsOut[offset], -i);
		}
		if (g_forwardMask[i] == 1){
			offset = g_scannedForwardMask[i];
			//g_countsOut[offset] += i+1;
			plus[offset] = i + 1;
			//atomicAdd(&g_countsOut[offset], i+1);
		}

	}
}


/*
__global__ void maskKernel(int *g_in, int* g_backwardMask, int* g_forwardMask, int n) {

	extern __shared__ int shared[];

	const int RADIUS = 1;
	const int BLOCK_SIZE = blockDim.x;

	for (int gindex : hemi::grid_stride_range(0, n)) {

		int lindex = threadIdx.x + RADIUS;

		shared[lindex] = g_in[gindex];

		if (threadIdx.x < RADIUS) {
			shared[lindex - RADIUS] = g_in[gindex - RADIUS];
			shared[lindex + BLOCK_SIZE] = g_in[gindex +BLOCK_SIZE];
		}

		__syncthreads();

		if (gindex == 0)
			g_backwardMask[gindex] = 1;
		else {
			// shared[lindex - 1]
			g_backwardMask[gindex] = (shared[lindex] != shared[lindex - 1]);
		}

		if (gindex == (n - 1))
			g_forwardMask[gindex] = 1;
		else {
			g_forwardMask[gindex] = (shared[lindex] !=	shared[lindex + 1]);
		}
		
		
	}
}

*/


// 70% memory

//unoptimized version.
__global__ void maskKernel(int *g_in, int* g_backwardMask, int* g_forwardMask, int n) {

	//__shared__ int

	for (int i : hemi::grid_stride_range(0, n)) {
		if (i == 0)
			g_backwardMask[i] = 1;
		else {
			g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
		}

		if (i == (n - 1))
			g_forwardMask[i] = 1;
		else {
			g_forwardMask[i] = (g_in[i] != g_in[i + 1]);
		}
	}
}



void PrintArray(int* arr, int n){
	for (int i = 0; i < n; ++i){
		printf("%d, ", arr[i]);
		/*if ( (i+1) % j == (0) && i!=0 ){
			printf("| ");
			}*/
	}
	printf("\n");
}

char errorString[256];

bool verifyCompression(
	int* original, int n,
	int* compressedSymbols, int* compressedCounts, int totalRuns){

	int* decompressed = new int[n];



	// decompress.
	int j = 0;

	int sum = 0;
	for (int i = 0; i < totalRuns; ++i) {
		sum += compressedCounts[i];
	}

	if (sum != n) {
		sprintf(errorString, "Decompressed and original size not equal %d != %d\n", n, sum);

		for (int i = 0; i < totalRuns; ++i){
			int symbol = compressedSymbols[i];
			int count = compressedCounts[i];

			printf("%d, %d\n", count, symbol);
		}

		return false;
	}


	for (int i = 0; i < totalRuns; ++i){
		int symbol = compressedSymbols[i];
		int count = compressedCounts[i];

		for (int k = 0; k < count; ++k){
			decompressed[j++] = symbol;
		}
	}

	// verify the compression.
	for (int i = 0; i < n; ++i) {
		if (original[i] != decompressed[i]){

			sprintf(errorString, "Decompressed and original not equal at %d, %d != %d\n", i, original[i], decompressed[i]);
			return false;
		}
	}


	return true;

	//printf("Decompressed:     ");
	//	PrintArray(decompressed, n);




}

int* getRandomData(int n){

	int* in = new int[n];

	int val = rand() % 10;

	for (int i = 0; i < n; ++i) {
		in[i] = val;

		if (rand() % 6 == 0){
			val = rand() % 10;
		}
	}

	return in;
}


template<typename F>
void unitTest(int* in, int n, F rle, bool verbose)
{

	int* symbolsOut = new int[n];
	int* countsOut = new int[n];

	int totalRuns = rle(in, n, symbolsOut, countsOut);
		//parleHost(in, n, symbolsOut, countsOut); // 1<<8

	if (verbose) {
		printf("n = %d\n", n);
		printf("Original Size  : %d\n", n);
		printf("Compressed Size: %d\n", totalRuns * 2);
	}

	if (!verifyCompression(
		in, n,
		symbolsOut, countsOut, totalRuns)) {
		printf("Failed test %s\n", errorString);
		PrintArray(in, n);

		exit(1);
	}
	else {
		if (verbose)
			printf("passed test!\n\n");
	}

	delete[] symbolsOut;
	delete[] countsOut;
}


template<typename F>
void profileCpu(F rle) {

	for (int i = 0; i < NUM_TESTS; ++i) {


		int n = Tests[i];
		
		int* in = getRandomData(n);
		int* symbolsOut = new int[n];
		int* countsOut = new int[n];


		for (int i = 0; i < TRIALS; ++i) {	

			sdkStartTimer(&timer);

			rle(in, n, symbolsOut, countsOut);

			sdkStopTimer(&timer);

		}

		// also unit test, to make sure that the compression is valid.
		unitTest(in, n, rle, false);


		printf("For n = %d, in time %.5f\n", n, sdkGetAverageTimerValue(&timer)*1e-3);
		

		delete[] in;
		delete[] symbolsOut;
		delete[] countsOut;

	}

}

template<typename F>
void profileGpu(bool useChag, F f) {

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	for (int i = 0; i < NUM_TESTS; ++i) {


		int n = Tests[i];

		int* in = getRandomData(n);
		int* symbolsOut = new int[n];
		int* countsOut = new int[n];




		int* d_symbolsOut;
		int* d_countsOut;
		int* d_in;
		int* d_totalRuns;

		int h_totalRuns;


		// MALLOC
		CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&d_countsOut, n * sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, n * sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&d_totalRuns, sizeof(int)));

		// transer input data to device.
		CUDA_CHECK(cudaMemcpy(d_in, in, n*sizeof(int), cudaMemcpyHostToDevice));

		cudaEventRecord(start);

		for (int i = 0; i < TRIALS; ++i) {

			// RUN.
			parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns, useChag);

			
		}


		cudaEventRecord(stop);
		cudaDeviceSynchronize();

		// FREE
		CUDA_CHECK(cudaFree(d_in));
		CUDA_CHECK(cudaFree(d_countsOut));
		CUDA_CHECK(cudaFree(d_symbolsOut));
		CUDA_CHECK(cudaFree(d_totalRuns));

		// also unit test, to make sure that the compression is valid.
		unitTest(in, n, f, false);

		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		printf("For n = %d, in time %.5f\n", n, (ms/((float)TRIALS ) ) /1000.0f);


		delete[] in;
		delete[] symbolsOut;
		delete[] countsOut;

	}

}


template<typename F>
void runTests(int a, F rle) {
	printf("START\n");

	
	for (int i = 4; i < a; ++i) {

		for (int k = 0; k < 30; ++k) {

			int n = 2 << i;

			if (k != 0) {
				// in first test, do with nice values for 'n'
				// on the other two tests, do with slightly randomized values.
				n = (int)(n * (0.6f + 1.3f * (rand() / (float)RAND_MAX)));
			}

			int* in = getRandomData(n);

			unitTest(in, n, rle, true);

			delete[] in;
		}

		printf("-------------------------------\n\n");
	}
}

int main()
{

	sdkCreateTimer(&timer);
	srand(1000);
	CUDA_CHECK(cudaSetDevice(0));


	cudppCreate(&cudpp);

	auto rleGpuCudpp = [](int *in, int n,
		int* symbolsOut,
		int* countsOut){
		return parleHost(in, n, symbolsOut, countsOut, false);
	};

	auto rleGpuChag = [](int *in, int n,
		int* symbolsOut,
		int* countsOut){
		return parleHost(in, n, symbolsOut, countsOut, true);
	};
	
	auto rleCpu = [](int *in, int n,
		int* symbolsOut,
		int* countsOut){
		return parleCpu(in, n, symbolsOut, countsOut);
	};

//	auto rle = rleGpu;

	
	/*
	
	int n = 260;
	int* in = new int[n]
	{
		
		9, 9, 9, 9, 9, 
		9, 9, 9, 9, 9, 
		9, 3, 3, 3, 3, 
		3, 3, 5, 5, 5, 
		5, 5, 5, 
		
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1,


		1, 1, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 5, 5, 5, 5, 1, 1, 7, 7, 7, 7,
			4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 8,
			8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 6, 6, 7
			, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 
			
			1, 1, 1, 1, 1,
			
			8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
			5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 5, 5, 5, 5, 
			
			5, 5, 5, 8, 5,
			5, 

			1, 1, 1, 1, 1, 1, 1, 1, 1, 

			4, 4, 7, 4, 4, 
			4, 6, 6, 6, 6, 
			6, 6, 8, 8, 8,
			8, 8, 7, 7, 7,		
			1, 1, 1, 2, 2, 

			
			
	}
	;

	unitTest(in, n, rleGpu, true); // 35



	delete[]in;
	*/
	
	
	

	// uni tests
	//runTests(21, rleGpuChag);

	//runTests(21, rleGpuCudpp);

	
	
//	printf("For CPU\n");
//	profileCpu(rleCpu);

	//printf("For GPU\n");
	//profileCpu(rleGpu);
	

	
	//printf("For GPU CUDPP\n");
	//profileGpu(false, rleGpuChag);
	

	//printf("For GPU CHAG\n");
	//profileGpu(true, rleGpuChag);
	


	//Visual Prof
	
	
	
	int n = 1 << 23;

	int* in = getRandomData(n);
	int* symbolsOut = new int[n];
	int* countsOut = new int[n];

	// also unit test, to make sure that the compression is valid.
	unitTest(in, n, rleGpuChag, true);


	delete[] in;
	delete[] symbolsOut;
	delete[] countsOut;



	CUDA_CHECK(cudaDeviceReset());


	printf("DONE\n");

	cudppDestroy(cudpp);

	return 0;
}


void scan(int* d_in, int* d_out, int N, bool useChag) {

	if (!useChag) {
		CUDPPConfiguration config;
		config.op = CUDPP_ADD;
		config.datatype = CUDPP_INT;
		config.algorithm = CUDPP_SCAN;
		config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

		CUDPPHandle plan = 0;
		CUDPPResult res = cudppPlan(cudpp, &plan, config, N, 1, 0);

		if (CUDPP_SUCCESS != res){
			printf("Error creating CUDPPPlan for scan!\n");
			exit(-1);
		}

		res = cudppScan(plan, d_out, d_in, N);
		if (CUDPP_SUCCESS != res){
			printf("Error in cudppScan()\n");
			exit(-1);
		}

		res = cudppDestroyPlan(plan);
		if (CUDPP_SUCCESS != res)
		{
			printf("Error destroying CUDPPPlan for scan\n");
			exit(-1);
		}
	}
	else {
		pp::prefix(d_in, d_in + N, d_out);
	}

}

void reduce(int* d_in, int* d_out, int N, bool useChag) {

	if (!useChag) {

		CUDPPConfiguration config;
		config.op = CUDPP_ADD;
		config.datatype = CUDPP_INT;
		config.algorithm = CUDPP_REDUCE;
		config.options = 0;

		CUDPPHandle plan = 0;
		CUDPPResult res = cudppPlan(cudpp, &plan, config, N, 1, 0);

		if (CUDPP_SUCCESS != res){
			printf("Error creating CUDPPPlan for reduce!\n");
			exit(-1);
		}

		res = cudppReduce(plan, d_out, d_in, N);
		if (CUDPP_SUCCESS != res){
			printf("Error in cudppReduce()\n");
			exit(-1);
		}

		res = cudppDestroyPlan(plan);
		if (CUDPP_SUCCESS != res)
		{
			printf("Error destroying CUDPPPlan for reduce\n");
			exit(-1);
		}

	}else{

		pp::reduce(d_in, d_in + N, d_out);


	}
}

int parleCpu(int *in, int n,
	int* symbolsOut,
	int* countsOut){

	if (n == 0)
		return 0; // nothing to compress!


	int outIndex = 0;
	int symbol = in[0];
	int count = 1;

	for (int i = 1; i < n; ++i) {

		if (in[i] != symbol) {
			// run is over.

			// So output run.
			symbolsOut[outIndex] = symbol;
			countsOut[outIndex] = count;
			outIndex++;

			// and start new run:
			symbol = in[i];
			count = 1;
		}
		else {
			// run is not over yet.
			++count;
		}

	}

	if (count > 0) {
		// output last run. 
		symbolsOut[outIndex] = symbol;
		countsOut[outIndex] = count;
	}

	return outIndex+1;

}

int parleHost(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut,
	bool useChag){

	int* d_symbolsOut;
	int* d_countsOut;
	int* d_in;
	int* d_totalRuns;

	int h_totalRuns;

	
	/*
	printf("N: %d\n", n);
	printf("n: %d\n", n);
	
	printf("orig:             ");
	PrintArray(h_in, n);
	*/

	// MALLOC
	CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_countsOut, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_totalRuns, sizeof(int)));

	// transer input data to device.
	CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice));


	// RUN.
	parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns, useChag);

	// MEMCPY
	CUDA_CHECK(cudaMemcpy(h_symbolsOut, d_symbolsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_countsOut, d_countsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));

	// FREE
	CUDA_CHECK(cudaFree(d_in));
	CUDA_CHECK(cudaFree(d_countsOut));
	CUDA_CHECK(cudaFree(d_symbolsOut));
	CUDA_CHECK(cudaFree(d_totalRuns));

	return h_totalRuns;

}

void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns, // keeps track of the total number of runs that the data was compressed down to.
	bool useChag
	){

	int* d_backwardMask;
	int* d_scannedBackwardMask;
	int* d_scannedForwardMask;
	int* d_forwardMask;
	int* d_plus;
	int* d_minus;

	// allocate resources on device. 
	CUDA_CHECK(cudaMalloc((void**)&d_backwardMask, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_forwardMask, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedBackwardMask, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedForwardMask, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_minus, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_plus, n * sizeof(int)));


	/*
	const int BLOCK_SIZE = 256;
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(BLOCK_SIZE); // TODO: compute this using occupancy API.
	ep.setSharedMemBytes((BLOCK_SIZE + 2)*sizeof(int));


	// get forward and backward mask. 
	hemi::cudaLaunch(ep, maskKernel, d_in, d_backwardMask, d_forwardMask, n);
	*/
	hemi::cudaLaunch(maskKernel, d_in, d_backwardMask, d_forwardMask, n);

	scan(d_backwardMask, d_scannedBackwardMask, n, useChag);
	scan(d_forwardMask, d_scannedForwardMask, n, useChag);

	reduce(d_backwardMask, d_totalRuns, n, useChag);

	hemi::cudaLaunch(scatterKernel, d_backwardMask, d_scannedBackwardMask,
		d_forwardMask, d_scannedForwardMask,
		d_in,
		d_symbolsOut, d_countsOut, n, d_plus, d_minus);


	hemi::cudaLaunch(sumKernel, d_countsOut, n, d_plus, d_minus);

	/*
	int* h_backwardMask = new int[n];
	int* h_forwardMask = new int[n];
	int* h_scannedBackwardMask = new int[n];
	int* h_scannedForwardMask = new int[n];
	int* h_plus = new int[n];
	int* h_minus = new int[n];

	

	CUDA_CHECK(cudaMemcpy(h_backwardMask, d_backwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_forwardMask, d_forwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(h_scannedBackwardMask, d_scannedBackwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_scannedForwardMask, d_scannedForwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	
	CUDA_CHECK(cudaMemcpy(h_plus, d_plus, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_minus, d_minus, n*sizeof(int), cudaMemcpyDeviceToHost));

	
	printf("Backward:         ");
	PrintArray(h_backwardMask, n);

	printf("Forward:          ");
	PrintArray(h_forwardMask, n);


	printf("Scanned Backward: ");
	PrintArray(h_scannedBackwardMask, n);

	printf("Scanned Forward:  ");
	PrintArray(h_scannedForwardMask, n);
	
	//printf("h_totalRuns:      %d\n", h_totalRuns);

	//printf("h_symbolsOut:     ");
	//PrintArray(h_symbolsOut, h_totalRuns);

	//printf("h_countsOut:      ");
	//PrintArray(h_countsOut, h_totalRuns);

	//printf("h_plus:     ");
	//PrintArray(h_plus, h_totalRuns);

	//printf("h_minus:      ");
	//PrintArray(h_minus, h_totalRuns);
	





	delete[] h_backwardMask;
	delete[] h_forwardMask;
	delete[] h_scannedBackwardMask;
	delete[] h_scannedForwardMask;

	*/
	

	CUDA_CHECK(cudaFree(d_backwardMask));
	CUDA_CHECK(cudaFree(d_forwardMask));
	CUDA_CHECK(cudaFree(d_scannedBackwardMask));
	CUDA_CHECK(cudaFree(d_scannedForwardMask));

	CUDA_CHECK(cudaFree(d_plus));
	CUDA_CHECK(cudaFree(d_minus));



}

/*
// Helper function for using CUDA to add vectors in parallel.
void scan(int *in, int* out, int n)
{
int *inBuffer = 0;
int *outBuffer = 0;

const unsigned int BLOCK_SIZE = 1024;


CUDA_CHECK(cudaMalloc((void**)&inBuffer, n * sizeof(int)));
CUDA_CHECK(cudaMalloc((void**)&outBuffer, n * sizeof(int)));

CUDA_CHECK(cudaMemcpy(inBuffer, in, n * sizeof(int), cudaMemcpyHostToDevice));

// warm up.
scanKernel<<<1, n, sizeof(int)*n*2 >> >( inBuffer, outBuffer, n);


for (int i = 0; i < 20; ++i){
cudaDeviceSynchronize();
sdkStartTimer(&timer);

scanKernel << <1, n, sizeof(int)*n * 2 >> >(inBuffer, outBuffer, n);

// Copy output vector from GPU buffer to host memory.
cudaDeviceSynchronize();
sdkStopTimer(&timer);
}

double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
printf("average: %.4f GB/s, Time = %.5f s\n", 1.0e-9 * ( (double)  (sizeof( int)*n)/reduceTime  ), reduceTime     );



CUDA_CHECK(cudaMemcpy(out, outBuffer, n * sizeof(int), cudaMemcpyDeviceToHost));

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

cudaFree(inBuffer);
cudaFree(outBuffer);
}
*/