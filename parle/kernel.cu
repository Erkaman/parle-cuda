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

bool improved = true;

const int MAX_N = 1 << 25; // max size of any array that we use.

int* g_symbolsOut;
int* g_countsOut;


int* g_in;

int* g_decompressed;


// used in PARLE. 
int* d_symbolsOut;
int* d_countsOut;
int* d_in;
int* d_totalRuns;

// used in parle as in-between storage arrays.
int* d_backwardMask;
int* d_scannedBackwardMask;
int* d_compactedBackwardMask;



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

__global__ void compactKernel(int* g_in, int* g_scannedBackwardMask, int* g_compactedBackwardMask, int* g_totalRuns, int n) {


	for (int i : hemi::grid_stride_range(0, n)) {

		if (i == (n - 1)) {
			g_compactedBackwardMask[g_scannedBackwardMask[i] + 0] = i + 1;

		//	printf("total runs in kernel %d\n", g_scannedBackwardMask[i]);

			*g_totalRuns = g_scannedBackwardMask[i];

		//	printf("total runs in kernel %d\n", *g_totalRuns);

		}

		if (i == 0) {
			g_compactedBackwardMask[0] = 0;
		}
		else if (g_scannedBackwardMask[i] != g_scannedBackwardMask[i-1]) {

			g_compactedBackwardMask[g_scannedBackwardMask[i] - 1] = i;

		}

	}
}


__global__ void scatterKernel(int* g_compactedBackwardMask, int* g_totalRuns, int* g_in, int* g_symbolsOut, int* g_countsOut) {

	int n = *g_totalRuns;

	for (int i : hemi::grid_stride_range(0, n)) {

		int a = g_compactedBackwardMask[i];
		int b = g_compactedBackwardMask[i+1];

		g_symbolsOut[i] = g_in[a];
		g_countsOut[i] = b-a;
	}

}


__global__ void maskKernel(int *g_in, int* g_backwardMask, int n) {

	extern __shared__ int shared[];

	const int RADIUS = 1;
	const int BLOCK_SIZE = blockDim.x;

	for (int gindex : hemi::grid_stride_range(0, n)) {

		int lindex = threadIdx.x + RADIUS;

		shared[lindex] = g_in[gindex];

		if (threadIdx.x < RADIUS) {
			shared[lindex - RADIUS] = g_in[gindex - RADIUS];
		}

		__syncthreads();

		if (gindex == 0)
			g_backwardMask[gindex] = 1;
		else {
			g_backwardMask[gindex] = (shared[lindex] != shared[lindex - 1]);
		}

	}
}



/*
__global__ void maskKernel(int *g_in, int* g_backwardMask, int n) {

	for (int i : hemi::grid_stride_range(0, n)) {
		if (i == 0)
			g_backwardMask[i] = 1;
		else {
			g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
		}
	}
}

*/

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
			g_decompressed[j++] = symbol;
		}
	}

	// verify the compression.
	for (int i = 0; i < n; ++i) {
		if (original[i] != g_decompressed[i]){

			sprintf(errorString, "Decompressed and original not equal at %d, %d != %d\n", i, original[i], g_decompressed[i]);
			return false;
		}
	}


	return true;

	//printf("Decompressed:     ");
	//	PrintArray(decompressed, n);




}

int* getRandomData(int n){


	int val = rand() % 10;

	for (int i = 0; i < n; ++i) {
		g_in[i] = val;

		if (rand() % 6 == 0){
			val = rand() % 10;
		}
	}

	return g_in;
}


template<typename F>
void unitTest(int* in, int n, F rle, bool verbose)
{

	
	int totalRuns = rle(in, n, g_symbolsOut, g_countsOut);
		//parleHost(in, n, symbolsOut, countsOut); // 1<<8

	if (verbose) {
		printf("n = %d\n", n);
		printf("Original Size  : %d\n", n);
		printf("Compressed Size: %d\n", totalRuns * 2);
	}

	if (!verifyCompression(
		in, n,
		g_symbolsOut, g_countsOut, totalRuns)) {
		printf("Failed test %s\n", errorString);
		PrintArray(in, n);

		exit(1);
	}
	else {
		if (verbose)
			printf("passed test!\n\n");
	}

}


template<typename F>
void profileCpu(F rle) {

	for (int i = 0; i < NUM_TESTS; ++i) {


		int n = Tests[i];
		
		int* in = getRandomData(n);
		

		for (int i = 0; i < TRIALS; ++i) {	

			sdkStartTimer(&timer);

			rle(in, n, g_symbolsOut, g_countsOut);

			sdkStopTimer(&timer);

		}

		// also unit test, to make sure that the compression is valid.
		unitTest(in, n, rle, false);


		printf("For n = %d, in time %.5f\n", n, sdkGetAverageTimerValue(&timer)*1e-3);
		


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
		



		
		int h_totalRuns;



		// transer input data to device.
		CUDA_CHECK(cudaMemcpy(d_in, in, n*sizeof(int), cudaMemcpyHostToDevice));

		cudaEventRecord(start);

		for (int i = 0; i < TRIALS; ++i) {

			// RUN.
			parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns, useChag);

			
		}


		cudaEventRecord(stop);
		cudaDeviceSynchronize();

		
		// also unit test, to make sure that the compression is valid.
		unitTest(in, n, f, false);

		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		printf("For n = %d, in time %.5f\n", n, (ms/((float)TRIALS ) ) /1000.0f);


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

	// allocate resources on device. These arrays are used globally thoughouts the program.
	CUDA_CHECK(cudaMalloc((void**)&d_backwardMask, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedBackwardMask, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_compactedBackwardMask, (MAX_N+ 1) * sizeof(int)));

	// MALLOC
	CUDA_CHECK(cudaMalloc((void**)&d_in, MAX_N* sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_countsOut, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_totalRuns, sizeof(int)));


	g_in = new int[MAX_N];
	g_decompressed = new int[MAX_N];

	g_symbolsOut = new int[MAX_N];
	g_countsOut = new int[MAX_N];



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
	int n = 8;
	int* in = new int[n]
	{
	//	1,2,3,4,4,4,2,2

	//	1,1,2, 4,4,4,4,4

		5,5,5,5,5,   1,2,3
	}
	;

	unitTest(in, n, rleGpuChag, true); // 35
	delete[]in;
	*/
	

	

	// uni tests
	//runTests(21, rleGpuChag);

	//runTests(21, rleGpuCudpp);
	
	//runTests(21, rleCpu);

	

	
	
	//printf("For CPU\n");
	//profileCpu(rleCpu);

	//rintf("For GPU\n");
	//profileCpu(rleGpu);



	
	//printf("For GPU CHAG\n");
	//profileGpu(true, rleGpuChag);
	
	//printf("For GPU CHAG\n");
	//profileGpu(true, rleGpuChag);
	


	//Visual Prof
	
	

	
	int n = 1 << 23;
	// also unit test, to make sure that the compression is valid.
	unitTest(getRandomData(1<<23), n, rleGpuChag, true);




	// free device arrays.
	CUDA_CHECK(cudaFree(d_backwardMask));
	CUDA_CHECK(cudaFree(d_scannedBackwardMask));
	CUDA_CHECK(cudaFree(d_compactedBackwardMask));

	// FREE
	CUDA_CHECK(cudaFree(d_in));
	CUDA_CHECK(cudaFree(d_countsOut));
	CUDA_CHECK(cudaFree(d_symbolsOut));
	CUDA_CHECK(cudaFree(d_totalRuns));


	CUDA_CHECK(cudaDeviceReset());

	// free global memory.
	delete[] g_in;
	delete[] g_decompressed;

	delete[] g_symbolsOut;
	delete[] g_countsOut;



	printf("DONE\n");

	cudppDestroy(cudpp);

	return 0;
}

void scan2(int* d_in, int* d_out, int N, bool useChag) {

		CUDPPConfiguration config;
		config.op = CUDPP_ADD;
		config.datatype = CUDPP_INT;
		config.algorithm = CUDPP_SCAN;
		config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

		CUDPPHandle plan = 0;
		CUDPPResult res = cudppPlan(cudpp, &plan, config, N, 1, 0);

		if (CUDPP_SUCCESS != res){
			printf("Error creating CUDPPPlan for scan2!\n");
			exit(-1);
		}

		res = cudppScan(plan, d_out, d_in, N);
		if (CUDPP_SUCCESS != res){
			printf("Error in cudppScan2()\n");
			exit(-1);
		}

		res = cudppDestroyPlan(plan);
		if (CUDPP_SUCCESS != res)
		{
			printf("Error destroying CUDPPPlan for scan2\n");
			exit(-1);
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

	int h_totalRuns;

	
	/*
	printf("N: %d\n", n);
	printf("n: %d\n", n);
	
	printf("orig:             ");
	PrintArray(h_in, n);
	*/

	
	// transer input data to device.
	CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice));


	// RUN.	
	parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns, useChag);

	// MEMCPY
	CUDA_CHECK(cudaMemcpy(h_symbolsOut, d_symbolsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_countsOut, d_countsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));

	//printf("total runs: %d\n", h_totalRuns);

	return h_totalRuns;

}


void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns, // keeps track of the total number of runs that the data was compressed down to.
	bool useChag
	){

	
	
	
	const int BLOCK_SIZE = 1024;
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(BLOCK_SIZE); // TODO: compute this using occupancy API.
	ep.setSharedMemBytes((BLOCK_SIZE + 1)*sizeof(int));
	
	hemi::cudaLaunch(ep, maskKernel, d_in, d_backwardMask, n);
	
	scan2(d_backwardMask, d_scannedBackwardMask, n, useChag);
	
	hemi::cudaLaunch(compactKernel, d_in, d_scannedBackwardMask, d_compactedBackwardMask, d_totalRuns, n);


	
	hemi::cudaLaunch(scatterKernel, d_compactedBackwardMask, d_totalRuns, d_in, d_symbolsOut, d_countsOut);
	


	/*
	int* h_backwardMask = new int[n];
	int* h_scannedBackwardMask = new int[n];
	int* h_compactedBackwardMask = new int[n+1];
	

	CUDA_CHECK(cudaMemcpy(h_backwardMask, d_backwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_scannedBackwardMask, d_scannedBackwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	
	CUDA_CHECK(cudaMemcpy(h_compactedBackwardMask, d_compactedBackwardMask, (n+1)*sizeof(int), cudaMemcpyDeviceToHost));


	printf("Backward:         ");
	PrintArray(h_backwardMask, n);

	printf("Scanned Backward: ");
	PrintArray(h_scannedBackwardMask, n);

	//printf("h_totalRuns:      %d\n", h_totalRuns);

	//printf("h_symbolsOut:     ");
	//PrintArray(h_symbolsOut, h_totalRuns);

	//printf("h_countsOut:      ");
	//PrintArray(h_countsOut, h_totalRuns);

	printf("d_compactedBackwardMask:     ");
	PrintArray(h_compactedBackwardMask, 10);


	delete[] h_backwardMask;
	delete[] h_scannedBackwardMask;
	delete[] h_compactedBackwardMask;
	*/

	


}


/*
todo:
organize the threadblocks so that we have coalesced read and writes.

think about occupancy. 

check the thoughput of a simple copy kernel in nvpp.

shrink thread blocks, to decrease waiting time at syncthreads. ug
*/