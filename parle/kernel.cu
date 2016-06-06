
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"
#include "helper_cuda.h"

void scan(int *in, int* out, int n);

void rle(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut);


__global__ void scanKernel(int *g_idata, int *g_odata, int n) {
	extern __shared__ float temp[]; // size is 2*n
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
	int *g_symbolsOut, int *g_countsOut, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	if (i < n){

		int globalOffset = g_blocksOffset[blockIdx.x];

		if (g_backwardMask[i] == 1){

			int localOffset = g_scannedBackwardMask[i];
			
			int symbol = g_in[i];

			g_symbolsOut[localOffset + globalOffset] = symbol;
			g_countsOut[localOffset + globalOffset] += -tid;
		}
		if (g_forwardMask[i] == 1){
			int localOffset = g_scannedForwardMask[i];
			g_countsOut[localOffset + globalOffset] += tid+1;
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

void PrintArray(int* arr, int n){
	for (int i = 0; i < n; ++i){
		printf("%d, ", arr[i]);
		if (i == (n/2-1) ){
			printf("| ");
		}
	}
	printf("\n");
}


int main()
{
	sdkCreateTimer(&timer);

	const int n = 16;

	int* in = new int[n];
	
	// 4, 4, 4, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 3, 1 

	int i = 0;
	in[i++] = 4; in[i++] = 4; in[i++] = 4;// 3
	in[i++] = 2; // 1
	in[i++] = 3; in[i++] = 3; in[i++] = 3; // 3
	in[i++] = 1; in[i++] = 1; in[i++] = 1; 	in[i++] = 9; in[i++] = 1; in[i++] = 1; // 6
	in[i++] = 2; // 1
	in[i++] = 3; // 1
	in[i++] = 1; // 1

	
	CUDA_CHECK(cudaSetDevice(0));


	int* symbolsOut = new int[2 * n];
	int* countsOut = new int[2 * n];

	rle(in, n, symbolsOut, countsOut);

	// input: 
	printf("Input:            ");
	PrintArray(in, n);
	
	CUDA_CHECK(cudaDeviceReset());


	printf("DONE\n");
	return 0;
}

void rle(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut){

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



	const int BLOCK_COUNT = 2;
	const int BLOCK_SIZE = n / BLOCK_COUNT;

	// allocate resources on device. 
	CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_backwardMask, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_forwardMask, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedBackwardMask, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedForwardMask, n * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_blocksRunCount, BLOCK_COUNT * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_blocksOffset, BLOCK_COUNT * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_countsOut, 2 * n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, 2 * n * sizeof(int)));


	// transer input data to device.
	CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice));

	// get forward and backward mask. 
	maskKernel<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_in, d_backwardMask, d_forwardMask, n);

	scanKernel << <BLOCK_COUNT, BLOCK_SIZE, 2 * n * sizeof(int) >> >(d_backwardMask, d_scannedBackwardMask, n);
	scanKernel << <BLOCK_COUNT, BLOCK_SIZE, 2 * n * sizeof(int) >> >(d_forwardMask, d_scannedForwardMask, n);


	getBlocksRunCountKernel << <BLOCK_COUNT, BLOCK_SIZE >> >(d_scannedBackwardMask, d_blocksRunCount, n);

	scanKernel << <1, BLOCK_COUNT, 2 * BLOCK_COUNT * sizeof(int) >> >(d_blocksRunCount, d_blocksOffset, n);



	scatterKernel << <BLOCK_COUNT, BLOCK_SIZE >> >(
		d_backwardMask, d_scannedBackwardMask, 
		d_forwardMask, d_scannedForwardMask,

		d_blocksOffset, d_in, 
		d_symbolsOut, d_countsOut, n);


	
	int* h_backwardMask        = new int[n];
	int* h_forwardMask         = new int[n];
	int* h_scannedBackwardMask = new int[n];
	int* h_scannedForwardMask  = new int[n];

	int* h_blocksRunCount      = new int[BLOCK_COUNT];
	int* h_blocksOffset        = new int[BLOCK_COUNT];

	CUDA_CHECK(cudaMemcpy(h_backwardMask, d_backwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_forwardMask, d_forwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_scannedBackwardMask, d_scannedBackwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_scannedForwardMask, d_scannedForwardMask, n*sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(h_blocksRunCount, d_blocksRunCount, BLOCK_COUNT*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_blocksOffset, d_blocksOffset, BLOCK_COUNT*sizeof(int), cudaMemcpyDeviceToHost));

	
	CUDA_CHECK(cudaMemcpy(h_symbolsOut, d_symbolsOut, 2*n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_countsOut, d_countsOut, 2 * n*sizeof(int), cudaMemcpyDeviceToHost));
	

	printf("Backward:         ");
	PrintArray(h_backwardMask, n);

	printf("Forward:          ");
	PrintArray(h_forwardMask, n);

	printf("Scanned Backward: ");
	PrintArray(h_scannedBackwardMask, n);

	printf("Scanned Forward:  ");
	PrintArray(h_scannedForwardMask, n);

	printf("h_blocksRunCount: ");
	PrintArray(h_blocksRunCount, BLOCK_COUNT);

	printf("h_blocksOffset:   ");
	PrintArray(h_blocksOffset, BLOCK_COUNT);

	printf("h_symbolsOut:     ");
	PrintArray(h_symbolsOut, 10);

	printf("h_countsOut:      ");
	PrintArray(h_countsOut, 10);

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