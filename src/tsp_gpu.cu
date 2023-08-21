#include "csr.cuh"
#include "kernels.cuh"
#include "helper_cuda.h"

TSP_GPU::TSP_GPU(TSP_CPU &tsp_cpu)
{
    initialize(tsp_cpu);
}

void TSP_GPU::initialize(TSP_CPU& tsp_cpu)
{
    ncities = tsp_cpu.ncities;
    ndistances = tsp_cpu.nonzeros;

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(1, 1);
    blocksPerGrid.x = ceil(double(ncities) / double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(ncities) / double(threadsPerBlock.y));
    cudaError_t error;

    cpu_distances = (double*)malloc(ndistances*sizeof(double));
    total_distances_cpu = (double*)malloc(ncities*sizeof(double));

    checkCudaErrors(cudaMalloc(&gpu_distances, ndistances*sizeof(double)));
    checkCudaErrors(cudaMalloc(&gpu_latitudes, ncities*sizeof(double)));
    checkCudaErrors(cudaMalloc(&gpu_longitudes, ncities*sizeof(double)));
    checkCudaErrors(cudaMalloc(&total_distances_gpu, ncities*sizeof(double)));

    checkCudaErrors(cudaMemcpy(gpu_latitudes, tsp_cpu.latitudes.data(), ncities*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_longitudes, tsp_cpu.longitudes.data(), ncities*sizeof(double), cudaMemcpyHostToDevice));

    if (tsp_cpu.dtype == "GEO") {
        TSP_distances_GEO<<<blocksPerGrid, threadsPerBlock>>>(gpu_distances, gpu_latitudes, gpu_longitudes, ncities);
    } else if (tsp_cpu.dtype == "EUC_2D") {
        TSP_distances_EUC_2D<<<blocksPerGrid, threadsPerBlock>>>(gpu_distances, gpu_latitudes, gpu_longitudes, ncities);
    }

    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error 0: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy(cpu_distances, gpu_distances, ndistances*sizeof(double), cudaMemcpyDeviceToHost));
    int i = 0;
    int j = 0;
    for (int k = 0; k < ncities; k++) {
        total_distances_cpu[k] = 0;
    }

    for (int k = 0; k < ndistances; k++) {
        if (j == ncities - 1) {
            i++;
            j = i;
        }
        j++;
        double d_ij = cpu_distances[k];
        total_distances_cpu[i] += d_ij;
        total_distances_cpu[j] += d_ij;
        sum_distance += d_ij;
        if (d_ij > maximum) {
            maximum = d_ij;
        }
    }

    std::cout << "\n";
    std::cout << maximum << "\n";

    checkCudaErrors(cudaMemcpy(total_distances_gpu, total_distances_cpu, ncities*sizeof(double), cudaMemcpyHostToDevice));
}

vector<double> TSP_GPU::construct_matrix() {

    // construct the TSP interaction matrix from A, B, C, and the distances calculated during the initialization of the TSP_GPU object

    vector<double> full_matrix;
	A = 1;
	B = maximum;
	C = maximum;
	for (int i = 0; i < ncities; i++) {
		for (int k = 0; k < ncities; k++) {
			for (int j = 0; j < ncities; j++) {
				for (int l = 0; l < ncities; l++) {
                    if (i != j && k == l) {
                        full_matrix.push_back(C/4);
                    }
                    else if ((i == j + 1 || j == i + 1 || (i == 0 && j == ncities-1) || (i == ncities-1 && j == 0)) && k != l) {
                        int city1 = std::min(k, l);
                        int city2 = std::max(k, l);
                        int position = ((2 * ncities - city1 - 3) * city1) / 2 + city2 -1;
                        full_matrix.push_back(A * cpu_distances[position] / 8);
                    }
                    else if (i == j && k != l) {
                        full_matrix.push_back(B/4);
                    }
                    else if (i == j && k == l) {
                        full_matrix.push_back((B+C)/4);
					}
                    else {
                        full_matrix.push_back(0);
                    }
				}
			}
		}
	}

    return full_matrix;
}

void TSP_GPU::print_h() {
    vector<double> full_matrix;
	int A = 1;
	double B = maximum;
	double C = maximum;
	for (int i = 0; i < ncities; i++) {
		for (int k = 0; k < ncities; k++) {
            full_matrix.push_back(-(A * total_distances_cpu[k] + (ncities-2) * (B + C)) / 2);
            std::cout << full_matrix.back();
            std::cout << ", ";
    	}
	}
}