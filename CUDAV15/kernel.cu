#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "utilities.h"
#include "parasort.h"

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <thread>

#include <boost\thread\barrier.hpp>
#include <boost\sort\sort.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <thrust\device_vector.h>

using namespace std;

void performKnapsackDynamicCudaCalculations(const string& dirName, vector<string>& fileNames);
void performKnapsackDynamicCPUCalculations(const string& dirName, vector<string>& fileNames);
void performKnapsackParallelDynamicCPUCalculations(const string& dirName, vector<string>& fileNames, unsigned threadCount);
cudaError_t knapsackCudaDynamic(int *output, const int *val, const int *wt, unsigned int n, unsigned int W);

void performKnapsackSortingCudaCalculations(const string& dirName, vector<string>& fileNames);
void performKnapsackSortingCPUCalculations(const string& dirName, vector<string>& fileNames, unsigned threadCount);

__device__ int maxi(int a, int b) {
	return (a > b) ? a : b;
}

__global__ void knapsackDynamicKernelPrepare(int *output, int n, int W) {
	int w = blockIdx.x * blockDim.x + threadIdx.x;
	if (w > W) return;

	output[w] = -1;
	if (w == 0)
		output[w] = 0;
}

__global__ void knapsackDynamicKernel(int *wt, int *val, int *output, int i, int n, int W) {
	int w = blockIdx.x * blockDim.x + threadIdx.x;
	if (w > W) return;
	int currentIndex = (i % 2)*(W + 1) + w;
	int previousIndex = ((i - 1) % 2)*(W + 1) + w;

	if (w - wt[i - 1] < 0 || output[previousIndex - wt[i - 1]] < 0)
		output[currentIndex] = output[previousIndex];
	else
		output[currentIndex] = maxi(val[i - 1] + output[previousIndex - wt[i - 1]], output[previousIndex]);
	__syncthreads();
}

int main() {
	//generateRandomDataFile("myDataSet1", 100000, 5000000);
	//generateRandomDataFile("myDataSet2", 1000000, 500000);
	//generateRandomDataFile("myDataSet3", 100000, 2000000);
	//generateRandomDataFile("myDataSet4", 1000000, 2000000);
	vector<string> lowDimensional, largeScale, hugeScale;
	read_directory("low_dimensional", lowDimensional);
	read_directory("large_scale", largeScale);
	read_directory("huge_scale", hugeScale);

	std::cout << "===DANE MALEJ SKALI - PODEJSCIE DYNAMICZNE - CUDA===" << endl;
	performKnapsackDynamicCudaCalculations("low_dimensional", lowDimensional);
	std::cout << endl << "===DANE DUZEJ SKALI - PODEJSCIE DYNAMICZNE - CUDA===" << endl;
	performKnapsackDynamicCudaCalculations("large_scale", largeScale);
	std::cout << endl << "===W£ASNE DANE DUZEJ SKALI - PODEJSCIE DYNAMICZNE - CUDA===" << endl;
	performKnapsackDynamicCudaCalculations("huge_scale", hugeScale);

	std::cout << "===DANE MALEJ SKALI - PODEJSCIE DYNAMICZNE - CPU===" << endl;
	performKnapsackDynamicCPUCalculations("low_dimensional", lowDimensional);
	std::cout << endl << "===DANE DUZEJ SKALI - PODEJSCIE DYNAMICZNE - CPU===" << endl;
	performKnapsackDynamicCPUCalculations("large_scale", largeScale);
	//std::cout << endl << "===W£ASNE DANE DUZEJ SKALI - PODEJSCIE DYNAMICZNE - CPU===" << endl;
	//performKnapsackDynamicCPUCalculations("huge_scale", hugeScale);

	for (unsigned i = 2; i <= 4; i *= 2) {
		std::cout << "===DANE MALEJ SKALI - PODEJSCIE DYNAMICZNE - CPU "<< i << "===" << endl;
		performKnapsackParallelDynamicCPUCalculations("low_dimensional", lowDimensional, i);
		std::cout << endl << "===DANE DUZEJ SKALI - PODEJSCIE DYNAMICZNE - CPU " << i << "===" << endl;
		performKnapsackParallelDynamicCPUCalculations("large_scale", largeScale, i);
	}
	for (unsigned i = 4; i <= 4; i *= 2) {
		std::cout << endl << "===W£ASNE DANE DUZEJ SKALI - PODEJSCIE DYNAMICZNE - CPU " << i << "===" << endl;
		performKnapsackParallelDynamicCPUCalculations("huge_scale", hugeScale, i);
	}

	std::cout << endl << "===DANE MALEJ SKALI - PODEJSCIE APROKSYMACYJNE - CUDA===" << endl;
	performKnapsackSortingCudaCalculations("low_dimensional", lowDimensional);
	std::cout << endl << "===DANE DUZEJ SKALI - PODEJSCIE APROKSYMACYJNE - CUDA===" << endl;
	performKnapsackSortingCudaCalculations("large_scale", largeScale);
	std::cout << endl << "===W£ASNE DANE DUZEJ SKALI - PODEJSCIE APROKSYMACYJNE - CUDA===" << endl;
	performKnapsackSortingCudaCalculations("huge_scale", hugeScale);

	for (unsigned i = 1; i <= 4; i *= 2) {
		std::cout << endl << "===DANE MALEJ SKALI - PODEJSCIE APROKSYMACYJNE - CPU " << i << "===" << endl;
		performKnapsackSortingCPUCalculations("low_dimensional", lowDimensional, i);
		std::cout << endl << "===DANE DUZEJ SKALI - PODEJSCIE APROKSYMACYJNE - CPU " << i << "===" << endl;
		performKnapsackSortingCPUCalculations("large_scale", largeScale, i);
		std::cout << endl << "===W£ASNE DANE DUZEJ SKALI - PODEJSCIE APROKSYMACYJNE - CPU " << i << "===" << endl;
		performKnapsackSortingCPUCalculations("huge_scale", hugeScale, i);
	}
	
	system("pause");
	return 0;
}

void performKnapsackSortingCudaCalculations(const string& dirName, vector<string>& fileNames) {
	std::cout << StringPadding("file", 25) << StringPadding("n", 8) << StringPadding("W", 10)
		<< StringPadding("time(ms)", 14) << StringPadding("expected", 10) << StringPadding("obtained", 10)
		<< StringPadding("error(\%)", 10) << endl;
	for (auto it = fileNames.begin(); it != fileNames.end(); it++) {
		unsigned int n, W, expectedResult;
		int *values, *weights;

		auto ret = loadData(dirName, (*it), n, W, expectedResult);
		values = ret.first;
		weights = ret.second;

		std::cout << StringPadding((*it), 25) << StringPadding(to_string(n), 8) << StringPadding(to_string(W), 10);
		
		auto start = std::chrono::system_clock::now();

		thrust::device_vector<int> dev_values(values, values + n);
		thrust::device_vector<int> dev_weights(weights, weights + n);
		thrust::device_vector<float> dev_output(n);
		thrust::device_vector<int> indexes(n);
		thrust::transform(dev_values.begin(), dev_values.end(), dev_weights.begin(), dev_output.begin(),
			thrust::divides<float>());
		thrust::sequence(indexes.begin(), indexes.end());
		thrust::sort_by_key(dev_output.begin(), dev_output.end(), indexes.begin(), thrust::greater<float>());
		thrust::host_vector<int> h_indexes(indexes);
		
		unsigned int weight = 0, maxValue = 0;
		for (auto it2 = h_indexes.begin(); it2 != h_indexes.end(); it2++) {
			if (weight + weights[*it2] <= W) {
				weight += weights[*it2];
				maxValue += values[*it2];
			}
		}

		auto end = std::chrono::system_clock::now();
		auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();

		std::cout << StringPadding(to_string(elapsed), 14);
		std::cout << StringPadding(to_string(expectedResult), 10) << StringPadding(to_string(maxValue), 10) << StringPadding(to_string(((float)((int)expectedResult - maxValue) / (float)expectedResult)*100.0), 10) << std::endl;
	}
}

bool wayToSort(pair<float, int> i, pair<float, int> j) { return i.first > j.first; }

void performKnapsackSortingCPUCalculations(const string& dirName, vector<string>& fileNames, unsigned threadCount) {
	std::cout << StringPadding("file", 25) << StringPadding("n", 8) << StringPadding("W", 10)
		<< StringPadding("time(ms)", 14) << StringPadding("expected", 10) << StringPadding("obtained", 10)
		<< StringPadding("error(\%)", 10) << endl;
	for (auto it = fileNames.begin(); it != fileNames.end(); it++) {
		unsigned int n, W, expectedResult;
		int *values, *weights;

		auto ret = loadData(dirName, (*it), n, W, expectedResult);
		values = ret.first;
		weights = ret.second;

		std::cout << StringPadding((*it), 25) << StringPadding(to_string(n), 8) << StringPadding(to_string(W), 10);

		auto start = std::chrono::system_clock::now();

		pair<float, int> *output = new pair<float, int>[n];
		for (int i = 0; i < n; ++i) {
			output[i] = pair<float, int>(float(values[i]) / float(weights[i]), i);
		}
		if (threadCount == 1) {
			std::sort(output, output + n, wayToSort);
		}
		else {
			parasort(n, output, threadCount);
			std::reverse(output, output + n);
		}

		unsigned int weight = 0, maxValue = 0;
		for (auto i = 0; i < n; ++i) {
			//cout << output[i].first << " ";
			if (weight + weights[output[i].second] <= W) {
				weight += weights[output[i].second];
				maxValue += values[output[i].second];
			}
		}

		auto end = std::chrono::system_clock::now();
		auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();

		std::cout << StringPadding(to_string(elapsed), 14);
		std::cout << StringPadding(to_string(expectedResult), 10) << StringPadding(to_string(maxValue), 10) << StringPadding(to_string(((float)((int)expectedResult - maxValue) / (float)expectedResult)*100.0), 10) << std::endl;
	}
}

void performKnapsackDynamicCudaCalculations(const string& dirName, vector<string>& fileNames) {
	std::cout << StringPadding("file", 25) << StringPadding("n", 8) << StringPadding("W", 10)
		<< StringPadding("time(ms)", 14) << StringPadding("expected", 10) << StringPadding("obtained", 10)
		<< StringPadding("error(\%)", 10) << endl;
	for (auto it = fileNames.begin(); it != fileNames.end(); it++) {
		unsigned int n, W, expectedResult;
		int *values, *weights;

		auto ret = loadData(dirName, (*it), n, W, expectedResult);
		values = ret.first;
		weights = ret.second;

		int *output = new int[2 * (W + 1) * sizeof(int)];
		std::cout << StringPadding((*it), 25) << StringPadding(to_string(n), 8) << StringPadding(to_string(W), 10);

		auto start = std::chrono::system_clock::now();

		cudaError_t cudaStatus = knapsackCudaDynamic(output, values, weights, n, W);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "knapsackCuda failed!");
			return;
		}

		int max = -1;
		for (int j = 0; j <= W; j++) {
			//std::cout << output[(n % 2)*W + j] << " ";
			if (max < output[(n % 2)*W + j]) {
				max = output[(n % 2)*W + j];
			}
		}

		auto end = std::chrono::system_clock::now();
		auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();

		std::cout << StringPadding(to_string(elapsed), 14);
		std::cout << StringPadding(to_string(expectedResult), 10) << StringPadding(to_string(max), 10) << StringPadding(to_string((int)expectedResult - max), 10) << std::endl;

		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return;
		}
		delete[] values;
		delete[] weights;
		delete[] output;
	}
}

void performKnapsackDynamicCPUCalculations(const string& dirName, vector<string>& fileNames) {
	std::cout << StringPadding("file", 25) << StringPadding("n", 8) << StringPadding("W", 10)
		<< StringPadding("time(ms)", 14) << StringPadding("expected", 10) << StringPadding("obtained", 10)
		<< StringPadding("error(\%)", 10) << endl;
	for (auto it = fileNames.begin(); it != fileNames.end(); it++) {
		unsigned int n, W, expectedResult;
		int *values, *weights;

		auto ret = loadData(dirName, (*it), n, W, expectedResult);
		values = ret.first;
		weights = ret.second;

		int *output = new int[2 * (W + 1) * sizeof(int)];
		std::cout << StringPadding((*it), 25) << StringPadding(to_string(n), 8) << StringPadding(to_string(W), 10);

		auto start = std::chrono::system_clock::now();

		for (int w = 0; w <= W; ++w) {
			output[w] = -1;

		}
		output[0] = 0;

		for (int i = 1; i <= n; ++i) {
			for (int w = 0; w < W + 1; ++w) {
				int currentIndex = (i % 2)*(W + 1) + w;
				int previousIndex = ((i - 1) % 2)*(W + 1) + w;

				if (w - weights[i - 1] < 0 || output[previousIndex - weights[i - 1]] < 0)
					output[currentIndex] = output[previousIndex];
				else
					output[currentIndex] = max(values[i - 1] + output[previousIndex - weights[i - 1]], output[previousIndex]);
			}
		}

		int max = -1;
		for (int j = 0; j <= W; j++) {
			//std::cout << output[(n % 2)*W + j] << " ";
			if (max < output[(n % 2)*W + j]) {
				max = output[(n % 2)*W + j];
			}
		}

		auto end = std::chrono::system_clock::now();
		auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();

		std::cout << StringPadding(to_string(elapsed), 14);
		std::cout << StringPadding(to_string(expectedResult), 10) << StringPadding(to_string(max), 10) << StringPadding(to_string((int)expectedResult - max), 10) << std::endl;

		delete[] values;
		delete[] weights;
		delete[] output;
	}
}

boost::mutex io_mutex;

void dynamicCPUThread(boost::barrier &b, int* values, int* weights, int* output, const unsigned &n, const unsigned &W, const unsigned &threadCount, const int start, const int end) {
	for (int i = 1; i <= n; ++i) {
		//cout << i << " ";
		b.wait();
		for (int w = start; w < end; ++w) {
			int currentIndex = (i % 2)*(W + 1) + w;
			int previousIndex = ((i - 1) % 2)*(W + 1) + w;

			if (w - weights[i - 1] < 0 || output[previousIndex - weights[i - 1]] < 0)
				output[currentIndex] = output[previousIndex];
			else
				output[currentIndex] = max(values[i - 1] + output[previousIndex - weights[i - 1]], output[previousIndex]);
		}
	}
}

void performKnapsackParallelDynamicCPUCalculations(const string& dirName, vector<string>& fileNames, unsigned threadCount) {
	std::cout << StringPadding("file", 25) << StringPadding("n", 8) << StringPadding("W", 10)
		<< StringPadding("time(ms)", 14) << StringPadding("expected", 10) << StringPadding("obtained", 10)
		<< StringPadding("error(\%)", 10) << endl;
	for (auto it = fileNames.begin(); it != fileNames.end(); it++) {
		unsigned int n, W, expectedResult;
		int *values, *weights;

		auto ret = loadData(dirName, (*it), n, W, expectedResult);
		values = ret.first;
		weights = ret.second;

		int *output = new int[2 * (W + 1) * sizeof(int)];
		std::cout << StringPadding((*it), 25) << StringPadding(to_string(n), 8) << StringPadding(to_string(W), 10);

		auto start = std::chrono::system_clock::now();

		for (int w = 0; w <= W; ++w) {
			output[w] = -1;

		}
		output[0] = 0;

		vector<thread> threads;
		boost::barrier b(threadCount);
		//dynamicCPUThread(int* values, int* weights, int* output, int &i, int &W, int start, int end)
		for (int j = 0; j < threadCount-1; ++j) {
			thread t(&dynamicCPUThread, ref(b), values, weights, output, ref(n), ref(W), ref(threadCount),
				int(W/threadCount)*j, int(W / threadCount)*(j+1));
			//cout << "Starts " << int(W / threadCount)*j << "-" << int(W / threadCount)*(j + 1) << endl;
			threads.push_back(move(t));
		}
		thread t(&dynamicCPUThread, ref(b), values, weights, output, ref(n), ref(W), ref(threadCount),
			int(W / threadCount)*(threadCount - 1), int(W + 1));
		//cout << "Starts " << int(W / threadCount)*(threadCount - 1) << "-" << int(W + 2) << endl;
		threads.push_back(move(t));
		for (auto it = threads.begin(); it != threads.end(); ++it) {
			(*it).join();
		}

		int max = -1;
		for (int j = 0; j <= W; j++) {
			//std::cout << output[(n % 2)*W + j] << " ";
			if (max < output[(n % 2)*W + j]) {
				max = output[(n % 2)*W + j];
			}
		}

		auto end = std::chrono::system_clock::now();
		auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();

		std::cout << StringPadding(to_string(elapsed), 14);
		std::cout << StringPadding(to_string(expectedResult), 10) << StringPadding(to_string(max), 10) << StringPadding(to_string((int)expectedResult - max), 10) << std::endl;

		delete[] values;
		delete[] weights;
		delete[] output;
	}
}

cudaError_t knapsackCudaDynamic(int *output, const int *values, const int *weights, unsigned int n, unsigned int W) {
	int *dev_values = 0;
	int *dev_weights = 0;
	int *dev_output = 0;
	int i = 1;
	cudaError_t cudaStatus;

	int *h_output = 0;
	int *h_values = 0;
	int *h_weights = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMallocHost((void**)&h_output, 2 * (W + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1 failed!");
		goto Error;
	}

	cudaStatus = cudaMallocHost((void**)&h_values, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 2 failed!");
		goto Error;
	}

	memcpy(h_values, values, n * sizeof(int));

	cudaStatus = cudaMallocHost((void**)&h_weights, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 3 failed!");
		goto Error;
	}

	memcpy(h_weights, weights, n * sizeof(int));

	cudaStatus = cudaMalloc((void**)&dev_output, 2 * (W + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_values, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 2 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_weights, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 3 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_values, h_values, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 1 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_weights, h_weights, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 2 failed!");
		goto Error;
	}

	cudaEventRecord(start);
	knapsackDynamicKernelPrepare << <int((W + 1) / 1024) + 1, 1024 >> >(dev_output, n, W);
	while (i <= n) {
		knapsackDynamicKernel << <int((W + 1) / 1024) + 1, 1024 >> >(dev_weights, dev_values, dev_output, i, n, W);
		i++;
	}
	cudaEventRecord(stop);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "knapsackKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching knapsackKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(h_output, dev_output, 2 * (W + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 4 failed!");
		goto Error;
	}

	memcpy(output, h_output, 2 * (W + 1) * sizeof(int));

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//std::cout << "Execution Time : " << milliseconds << " milliseconds" << std::endl;

Error:
	cudaFree(dev_output);
	cudaFree(dev_values);
	cudaFree(dev_weights);
	cudaFreeHost(h_output);
	cudaFreeHost(h_values);
	cudaFreeHost(h_weights);

	return cudaStatus;
}