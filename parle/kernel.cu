#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"
#include "helper_cuda.h"

#include "hemi/grid_stride_range.h"
#include "hemi/launch.h"

#include "chag/pp/prefix.cuh"
#include "chag/pp/reduce.cuh"

namespace pp = chag::pp;

// global host memory arrays.
int* g_symbolsOut;
int* g_countsOut;
int* g_in;
int* g_decompressed;

// Device memory used in PARLE
int* d_symbolsOut;
int* d_countsOut;
int* d_in;
int* d_totalRuns;
int* d_backwardMask;
int* d_scannedBackwardMask;
int* d_compactedBackwardMask;

const int NUM_TESTS = 11;
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
	40000000,
};

const int PROFILING_TESTS = 100;
const int MAX_N = 1 << 26; // max size of any array that we use.

void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns
	);

int parleHost(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut);

int rleCpu(int *in, int n,
	int* symbolsOut,
	int* countsOut);

__global__ void compactKernel(int* g_in, int* g_scannedBackwardMask, int* g_compactedBackwardMask, int* g_totalRuns, int n) {
	for (int i : hemi::grid_stride_range(0, n)) {

		if (i == (n - 1)) {
			g_compactedBackwardMask[g_scannedBackwardMask[i] + 0] = i + 1;
			*g_totalRuns = g_scannedBackwardMask[i];
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
	for (int i : hemi::grid_stride_range(0, n)) {
		if (i == 0)
			g_backwardMask[i] = 1;
		else {
			g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
		}
	}
}

void PrintArray(int* arr, int n){
	for (int i = 0; i < n; ++i){
		printf("%d, ", arr[i]);
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
}

// get random test data for compression.
// the kind of data generated is like
// 1,1,1,1,4,4,4,4,7,7,7,7,....
// so there's lots of repeated sequences. 
int* generateCompressibleRandomData(int n){
	int val = rand() % 10;

	for (int i = 0; i < n; ++i) {
		g_in[i] = val;

		if (rand() % 6 == 0){
			val = rand() % 10;
		}
	}
	return g_in;
}


// get random test data for compression.
// the kind of data generated is like
// 1,5,8,4,2,6,....
// so it's completely random.
int* generateRandomData(int n){
	for (int i = 0; i < n; ++i) {
		g_in[i] = rand() % 10;;

	}
	return g_in;
}


// use f to RLE compresss the data, and then verify the compression. 
template<typename F>
void unitTest(int* in, int n, F f, bool verbose)
{
	int totalRuns = f(in, n, g_symbolsOut, g_countsOut);
	
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

// profile some RLE implementation on the CPU.
template<typename F, typename G>
void profileCpu(F rle, G dataGen) {
	for (int i = 0; i < NUM_TESTS; ++i) {
		int n = Tests[i];
		int* in = dataGen(n);
		
		for (int i = 0; i < PROFILING_TESTS; ++i) {
			sdkStartTimer(&timer);
			rle(in, n, g_symbolsOut, g_countsOut);
			sdkStopTimer(&timer);
		}

		// also unit test, to make sure that the compression is valid.
		unitTest(in, n, rle, false);

		printf("For n = %d, in time %.5f\n", n, sdkGetAverageTimerValue(&timer)*1e-3);
	}
}

// profile some RLE implementation on the GPU.
template<typename F, typename G>
void profileGpu(F rle, G dataGen) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < NUM_TESTS; ++i) {

		int n = Tests[i];
		int* in = dataGen(n);
		int h_totalRuns;

		// transer input data to device.
		CUDA_CHECK(cudaMemcpy(d_in, in, n*sizeof(int), cudaMemcpyHostToDevice));

		// record.
		cudaEventRecord(start);
		for (int i = 0; i < PROFILING_TESTS; ++i) {
			parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns);
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

// Run many unit tests on an implementation(f) of RLE.
template<typename F>
void runTests(int a, F f) {
	printf("START UNIT TESTS\n");

	for (int i = 4; i < a; ++i) {
		for (int k = 0; k < 30; ++k) {
			int n = 2 << i;

			if (k != 0) {
				// in first test, do with nice values for 'n'
				// on the other two tests, do with slightly randomized values.
				n = (int)(n * (0.6f + 1.3f * (rand() / (float)RAND_MAX)));
			}

			int* in = generateCompressibleRandomData(n);

			unitTest(in, n, f, true);
		}
		printf("-------------------------------\n\n");
	}
}

int main(){

	sdkCreateTimer(&timer);
	srand(1000);
	CUDA_CHECK(cudaSetDevice(0));

	// allocate resources on device. These arrays are used globally thoughouts the program.
	CUDA_CHECK(cudaMalloc((void**)&d_backwardMask, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedBackwardMask, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_compactedBackwardMask, (MAX_N+ 1) * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_in, MAX_N* sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_countsOut, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_totalRuns, sizeof(int)));

	// allocate resources on the host. 
	g_in = new int[MAX_N];
	g_decompressed = new int[MAX_N];
	g_symbolsOut = new int[MAX_N];
	g_countsOut = new int[MAX_N];

	// We run this code to run many unit tests on the code
	/*
	runTests(21, rleCpu);
	runTests(21, parleHost);
	*/

	// We run this code to profile the performance. 
	
	printf("profile CPU\n");
	profileCpu(rleCpu, generateRandomData);

	printf("profile1 GPU\n");
	profileCpu(parleHost, generateRandomData);
	
	printf("profile2 GPU\n");
	profileCpu(parleHost, generateCompressibleRandomData);

	
	
	// We run this code when we wish to run NVPP on the algorithm. 
	/*
	int n = 1 << 23;
    unitTest(generateCompressibleRandomData(1<<23), n, rleGpu, true);
	*/

	// free device arrays.
	CUDA_CHECK(cudaFree(d_backwardMask));
	CUDA_CHECK(cudaFree(d_scannedBackwardMask));
	CUDA_CHECK(cudaFree(d_compactedBackwardMask));
	CUDA_CHECK(cudaFree(d_in));
	CUDA_CHECK(cudaFree(d_countsOut));
	CUDA_CHECK(cudaFree(d_symbolsOut));
	CUDA_CHECK(cudaFree(d_totalRuns));

	CUDA_CHECK(cudaDeviceReset());

	// free host memory.
	delete[] g_in;
	delete[] g_decompressed;

	delete[] g_symbolsOut;
	delete[] g_countsOut;

	return 0;
}



// implementation of RLE on CPU.
int rleCpu(int *in, int n, int* symbolsOut, int* countsOut){

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
			++count; // run is not over yet.
		}
	}

	// output last run. 
	symbolsOut[outIndex] = symbol;	
	countsOut[outIndex] = count;
	outIndex++;

	return outIndex;
}

// On the CPU do preparation to run parle, launch PARLE on GPU, and then transfer the result data to the CPU. 
int parleHost(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut){

	int h_totalRuns;

	// transer input data to device.
	CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice));

	// RUN.	
	parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns);

	// transer result data to host.
	CUDA_CHECK(cudaMemcpy(h_symbolsOut, d_symbolsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_countsOut, d_countsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));

	return h_totalRuns;
}

void scan(int* d_in, int* d_out, int N) {
	pp::prefix_inclusive(d_in, d_in + N, d_out);
}

// run parle on the GPU
void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns
	){
	hemi::cudaLaunch(maskKernel, d_in, d_backwardMask, n);
	scan(d_backwardMask, d_scannedBackwardMask, n);
	hemi::cudaLaunch(compactKernel, d_in, d_scannedBackwardMask, d_compactedBackwardMask, d_totalRuns, n);
	hemi::cudaLaunch(scatterKernel, d_compactedBackwardMask, d_totalRuns, d_in, d_symbolsOut, d_countsOut);
}