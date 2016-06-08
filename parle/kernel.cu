
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"
#include "helper_cuda.h"

int parle(int *in, int n,
	int* h_symbolsOut,
	int* h_countsOut, int blockSize);


__global__ void scanKernel2(int *g_idata, int *g_odata, int n) {
	extern __shared__ int temp[]; // size is 2*n
	int tid = threadIdx.x;
	int pout = 0;
	int pin = 1;
	

	temp[pout*n + tid] = (tid > 0) ? g_idata[tid - 1] : 0;
	//temp[pout*n + i] = g_idata[i];

	__syncthreads();

	for (int offset = 1; offset < blockDim.x; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;
		if (tid >= offset)
			temp[pout*n + tid] = temp[pin*n + tid] + temp[pin*n + tid - offset];
		else
			temp[pout*n + tid] = temp[pin*n + tid];
		__syncthreads();
	}

	g_odata[tid] = temp[pout*n + tid]; // write output
}


// segmented scan that is run on each thread block. 
__global__ void scanKernel(int *g_idata, int *g_odata, int n) {
	extern __shared__ int temp[]; // size is 2*n
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;
	int pout = 0;
	int pin = 1;



	temp[pout*n + i] = (tid > 0) ? g_idata[i-1] : 0;
	//temp[pout*n + i] = g_idata[i];

	__syncthreads();
	
	for (int offset = 1; offset < blockDim.x; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;
		if (tid >= offset)
			temp[pout*n + i] = temp[pin*n + i] +  temp[pin*n + i - offset];
		else
			temp[pout*n + i] = temp[pin*n + i];
		__syncthreads();
	}
	
	g_odata[i] = temp[pout*n + i]; // write output
}

__global__ void getBlocksRunCountKernel(int *g_idata, int *g_odata, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	if (i < n){

		if (tid == (blockDim.x - 1)){
			g_odata[blockIdx.x] = g_idata[i]+1;
		}

	}
}

__global__ void scatterKernel(
	int *g_backwardMask, int* g_scannedBackwardMask, 
	int *g_forwardMask, int* g_scannedForwardMask,
	int *g_blocksOffset, int *g_in, 
	int *g_symbolsOut, int *g_countsOut, int n, int* g_totalRuns) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	if (i < n){

		int globalOffset = g_blocksOffset[blockIdx.x];
		int localOffset;

		if (g_backwardMask[i] == 1){

			localOffset = g_scannedBackwardMask[i];
			
			int symbol = g_in[i];

			g_symbolsOut[localOffset + globalOffset] = symbol;
			g_countsOut[localOffset + globalOffset] += -tid;
		}
		if (g_forwardMask[i] == 1){
			localOffset = g_scannedForwardMask[i];
			g_countsOut[localOffset + globalOffset] += tid+1;
		}

		if ((i + 1) == n) {
			*g_totalRuns = localOffset + globalOffset + 1;
		}



	}
}

__global__ void maskKernel(int *g_in, int* g_backwardMask, int* g_forwardMask, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	if (i < n){
		if (tid == 0) 
			g_backwardMask[i] = 1;
		else {
			g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
		}

		if (tid== (blockDim.x-1 ) )
			g_forwardMask[i] = 1;
		else {
			g_forwardMask[i] = (g_in[i] != g_in[i + 1]);
		}

	}
}

void PrintArray(int* arr, int n, int j){
	for (int i = 0; i < n; ++i){
		printf("%d, ", arr[i]);
		if ( (i+1) % j == (0) && i!=0 ){
			printf("| ");
		}
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
	for (int i = 0; i < totalRuns; ++i){
		int symbol = compressedSymbols[i];
		int count  = compressedCounts[i];

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

	int val = rand() % 1000;

	for (int i = 0; i < n; ++i) {
		in[i] = val;

		if (rand() % 6 == 0){
			val = rand() % 1000;
		}
	}

	return in;
}


void unitTest(int* in, int n, int blockSize)
{

	int* symbolsOut = new int[2 * n];
	int* countsOut = new int[2 * n];

	int totalRuns = parle(in, n, symbolsOut, countsOut, 30); // 1<<8

	printf("n = %d, blockSize = %d\n", n, blockSize);
	printf("Original Size  : %d\n", n);
	printf("Compressed Size: %d\n", totalRuns * 2);


	if (!verifyCompression(
		in, n,
		symbolsOut, countsOut, totalRuns)) {
		printf("Failed test %s\n", errorString);
		exit(1);
	}
	else {
		printf("passed test!\n\n");
	}

	delete[] symbolsOut;
	delete[] countsOut;
}


int main()
{
	sdkCreateTimer(&timer);
	srand(1000);
	CUDA_CHECK(cudaSetDevice(0));
	
	int n = 12;
	int* in = new int[12];

	int i = 0;
	in[i++] = 1; 
	in[i++] = 4;
	in[i++] = 4;
	in[i++] = 4;
	in[i++] = 2;
	in[i++] = 2;

	in[i++] = 2;
	in[i++] = 2;
	in[i++] = 2;
	in[i++] = 4;
	in[i++] = 5;
	in[i++] = 6;

	unitTest(in, n, 6);


	delete[]in;
	


	/*
	for (int i = 4; i < 12; ++i) {

		for (int j = 2; j < (i); ++j) {

			for (int k = 0; k < 3; ++k) {

				int n = 2 << i;
				int blockSize = 2 << j;

				if (k != 0) {
					// in first test, do with nice values for 'n' and 'blockSize'
					// on the other two tests, do with slightly randomized values.
					n += (-11 + rand() % 30);
					blockSize += (-3 + rand() % 11);

				}
				
				int* in = getRandomData(n);

				unitTest(in, n, blockSize);

				delete[] in;
			}


		}
		printf("-------------------------------\n\n");

	}
	*/
	
	
	


	
	CUDA_CHECK(cudaDeviceReset());
	

	printf("DONE\n");
	return 0;
}

int parle(int *in, int n,
	int* h_symbolsOut,
	int* h_countsOut,
	int blockSize){

	int* d_backwardMask;
	int* d_scannedBackwardMask;
	int* d_scannedForwardMask;
	int* d_forwardMask;
	int* d_in;



	// keeps track of the number of runs per block. So d_blocksRunCount[0] is the number of runs for block number 0. 
	int* d_blocksRunCount;
	int* d_blocksOffset;

	int* d_symbolsOut;

	int* d_countsOut;
	int* d_totalRuns; // keeps track of the total number of runs that the data was compressed down to.


	const int BLOCK_SIZE = blockSize;
	const int BLOCK_COUNT = (int)ceil( n / (double)BLOCK_SIZE );
	const int N = BLOCK_COUNT * BLOCK_SIZE;

	/*
	printf("N: %d\n", N);
	printf("n: %d\n", n);
	printf("blocksize: %d\n", BLOCK_SIZE);
	printf("BLOCK_COUNT: %d\n", BLOCK_COUNT);
	*/
	
	int padding = 0; // default padding char is 0.
	if (in[n - 1] == 0){
		padding = 1; // but else use 1. 
	}
	// we use padding if there is not enough input data to fill all the thread blocks. 
	bool usePadding = N != n; // 
	int* h_in = new int[N];

	for (int i = 0; i < N; ++i) {
		if (i < n){
			h_in[i] = in[i];
		}
		else{
			h_in[i] = padding; 
		}
	}
	
	/*
	printf("use pad: %d", usePadding);

	printf("orig:      ");
	PrintArray(in, n, BLOCK_SIZE);


	printf("padded:      ");	
	PrintArray(h_in, N, BLOCK_SIZE);
	*/


	// allocate resources on device. 
	CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_backwardMask, N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_forwardMask, N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedBackwardMask, N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedForwardMask, N * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_blocksRunCount, BLOCK_COUNT * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_blocksOffset, BLOCK_COUNT * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_countsOut, 2 * N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, 2 * N * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_totalRuns, sizeof(int)));


	// transer input data to device.
	CUDA_CHECK(cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice));

	
	// get forward and backward mask. 
	maskKernel<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_in, d_backwardMask, d_forwardMask, N);
	CUDA_CHECK(cudaGetLastError());
	
	
	scanKernel << <BLOCK_COUNT, BLOCK_SIZE, 2 * N * sizeof(int) >> >(d_backwardMask, d_scannedBackwardMask, N);
	CUDA_CHECK(cudaGetLastError());
	
	scanKernel << <BLOCK_COUNT, BLOCK_SIZE, 2 * N * sizeof(int) >> >(d_forwardMask, d_scannedForwardMask, N);
	CUDA_CHECK(cudaGetLastError());

	
	getBlocksRunCountKernel << <BLOCK_COUNT, BLOCK_SIZE >> >(d_scannedBackwardMask, d_blocksRunCount, N);
	CUDA_CHECK(cudaGetLastError());
	/*
	printf("block count:%d \n", BLOCK_COUNT);
	printf("allocate this muc:%d \n", 2 * BLOCK_COUNT * sizeof(int));
	*/
	// TODO: there may not be enough thread if there are many blocks!
	//scanKernel2 << <1, BLOCK_COUNT, 2 * BLOCK_COUNT * sizeof(int) >> >(d_blocksRunCount, d_blocksOffset, BLOCK_COUNT);
	scanKernel << <1, BLOCK_COUNT, 2 * BLOCK_COUNT * sizeof(int) >> >(d_blocksRunCount, d_blocksOffset, BLOCK_COUNT);

	CUDA_CHECK(cudaGetLastError());
	
	
	
	scatterKernel << <BLOCK_COUNT, BLOCK_SIZE >> >(
		d_backwardMask, d_scannedBackwardMask, 
		d_forwardMask, d_scannedForwardMask,

		d_blocksOffset, d_in, 
		d_symbolsOut, d_countsOut, N, d_totalRuns);
	CUDA_CHECK(cudaGetLastError());
	
	
	
	int* h_backwardMask        = new int[N];
	int* h_forwardMask         = new int[N];
	int* h_scannedBackwardMask = new int[N];
	int* h_scannedForwardMask  = new int[N];

	int* h_blocksRunCount      = new int[BLOCK_COUNT];
	int* h_blocksOffset        = new int[BLOCK_COUNT];

	int h_totalRuns;

	//CUDA_CHECK(cudaDeviceSynchronize());

	
	CUDA_CHECK(cudaMemcpy(h_backwardMask, d_backwardMask, N*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_forwardMask, d_forwardMask, N*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_scannedBackwardMask, d_scannedBackwardMask, N*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_scannedForwardMask, d_scannedForwardMask, N*sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(h_blocksRunCount, d_blocksRunCount, BLOCK_COUNT*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_blocksOffset, d_blocksOffset, BLOCK_COUNT*sizeof(int), cudaMemcpyDeviceToHost));
	
	
	CUDA_CHECK(cudaMemcpy(h_symbolsOut, d_symbolsOut, 2*n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_countsOut, d_countsOut, 2 *n*sizeof(int), cudaMemcpyDeviceToHost));
	
	CUDA_CHECK(cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));
	
	if (usePadding) {
		// if we use padding, then the last run will just be the compressed padding characters.
		// so skip that last run:
		--h_totalRuns;
	
	}
	
	
	/*
	printf("Backward:         ");
	PrintArray(h_backwardMask, N, BLOCK_SIZE);
	
	printf("Forward:          ");
	PrintArray(h_forwardMask, N, BLOCK_SIZE);

	printf("Scanned Backward: ");
	PrintArray(h_scannedBackwardMask, N, BLOCK_SIZE);

	printf("Scanned Forward:  ");
	PrintArray(h_scannedForwardMask, N, BLOCK_SIZE);
	
	printf("h_blocksRunCount: ");
	PrintArray(h_blocksRunCount, BLOCK_COUNT, BLOCK_SIZE);

	printf("h_blocksOffset:   ");
	PrintArray(h_blocksOffset, BLOCK_COUNT, BLOCK_SIZE);
	
	printf("h_symbolsOut:     ");
	PrintArray(h_symbolsOut, h_totalRuns,10000000);

	printf("h_countsOut:      ");
	PrintArray(h_countsOut, h_totalRuns,1000000);



	printf("h_totalRuns:      %d\n", h_totalRuns);
	*/

	delete []h_in;

	delete[] h_backwardMask;
	delete[] h_forwardMask;
	delete[] h_scannedBackwardMask;
	delete[] h_scannedForwardMask;

	delete[] h_blocksRunCount;
	delete[] h_blocksOffset;


	CUDA_CHECK(cudaFree(d_in));
	
	CUDA_CHECK(cudaFree(d_backwardMask));
	CUDA_CHECK(cudaFree(d_forwardMask));
	CUDA_CHECK(cudaFree(d_scannedBackwardMask));
	CUDA_CHECK(cudaFree(d_scannedForwardMask));

	CUDA_CHECK(cudaFree(d_blocksRunCount));
	CUDA_CHECK(cudaFree(d_blocksOffset));

	CUDA_CHECK(cudaFree(d_countsOut));
	CUDA_CHECK(cudaFree(d_symbolsOut));

	CUDA_CHECK(cudaFree(d_totalRuns));
	
	
	return h_totalRuns;
	// TODO: cudaFree.
	
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