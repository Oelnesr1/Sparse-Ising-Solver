#include "csr.cuh"
#include "kernels.cuh"
#include "helper_cuda.h"

void printSumDouble(double *cpu_pointer, double *gpu_pointer, int len, int i)
{
    cudaMemcpy(cpu_pointer, gpu_pointer, len * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int j = 0; j < len; j++)
    {
        sum += cpu_pointer[j];
    }

    std::cout << "sum of array at step " << i << ": " << sum << "\n";

}

void printSumInt(int *cpu_pointer, int *gpu_pointer, int len, int i)
{
        cudaMemcpy(cpu_pointer, gpu_pointer, len * sizeof(int), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int j = 0; j < len; j++)
    {
        sum += cpu_pointer[j];
    }
    std::cout << "\n";

    std::cout << "sum of array at step " << i << ": " << sum << "\n";

}

void printSumBool(bool *cpu_pointer, bool *gpu_pointer, int len, int i)
{
        cudaMemcpy(cpu_pointer, gpu_pointer, len * sizeof(bool), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int j = 0; j < len; j++)
    {
        sum += cpu_pointer[j];
    }

    std::cout << "sum of array at step " << i << ": " << sum << "\n";

}

Ising::Ising(CSR &csr_cpu, CSR_GPU &csr_gpu)
{
    initialize(csr_cpu, csr_gpu);
}

void Ising::initialize(CSR &csr_cpu, CSR_GPU &csr_gpu)
{
    nrows = csr_cpu.size;
    ncols = agents;
    SIZE = nrows * ncols;
    double_bytes = SIZE * sizeof(double);
    double_bytes_large = (nrows * nrows) * sizeof(double);
    int_bytes_large = SIZE * sizeof(int);
    int_bytes_small = nrows * sizeof(int);
    bool_bytes = ncols * sizeof(bool);

    X_cpu = (double *)malloc(double_bytes);
    Y_cpu = (double *)malloc(double_bytes);
    cut_array_cpu = (double*)malloc(double_bytes);
    final_spins_cpu = (int*)malloc(int_bytes_large);
    final_cut_array_cpu = (double*)malloc(ncols*sizeof(double));

    run = true;
    step = 0;

    checkCudaErrors(cudaMalloc(&step_gpu, sizeof(int)));
    checkCudaErrors(cudaMalloc(&X_gpu, double_bytes));
    checkCudaErrors(cudaMalloc(&Y_gpu, double_bytes));
    checkCudaErrors(cudaMalloc(&stability_gpu, int_bytes_large));

    checkCudaErrors(cudaMalloc(&cut_array_gpu, double_bytes));
    checkCudaErrors(cudaMalloc(&final_cut_array_gpu, ncols * sizeof(double)));

    checkCudaErrors(cudaMalloc(&current_spins_gpu, int_bytes_large));
    checkCudaErrors(cudaMalloc(&final_spins_gpu, int_bytes_large));

    checkCudaErrors(cudaMalloc(&bifurcated_gpu, bool_bytes));
    checkCudaErrors(cudaMalloc(&prev_bifurcated_gpu, bool_bytes));

}

void Ising::reset_initialization(CSR &csr_cpu, CSR_GPU &csr_gpu) {
    nrows = csr_cpu.size;
    ncols = agents;
    SIZE = nrows * ncols;

    for (int i = 0; i < SIZE; i++)
    {
        X_cpu[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        Y_cpu[i] = ((double)rand() / RAND_MAX) * 4 - 2;
    }

    double sum = 0.0f;
    double nonzero_contribution = 0.0f;
    double zero_contribution = 0.0f;

    for (int i = 0; i < csr_cpu.csr_data.size(); i++)
    {
        sum += csr_cpu.csr_data[i];
    }
    double mean = sum / (nrows * nrows);


    for( int i = 0; i < csr_cpu.nonzeros; i++) {
        nonzero_contribution += std::pow((csr_cpu.csr_data[i] - mean), 2);
    }

    zero_contribution = ((nrows * nrows) - csr_cpu.nonzeros) * mean * mean;

    sum = nonzero_contribution + zero_contribution;

    double variance = std::pow((sum / (nrows * nrows - 1)), 0.5);
    xi0 = 0.7 / (variance * std::pow(nrows, 0.5));

    checkCudaErrors(cudaMemset(step_gpu, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(current_spins_gpu, -1, int_bytes_large));
    checkCudaErrors(cudaMemset(final_spins_gpu, -1, int_bytes_large));
    checkCudaErrors(cudaMemset(bifurcated_gpu, 0, bool_bytes));
    checkCudaErrors(cudaMemset(prev_bifurcated_gpu, 0, bool_bytes));

    checkCudaErrors(cudaMemcpy(X_gpu, X_cpu, double_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Y_gpu, Y_cpu, double_bytes, cudaMemcpyHostToDevice));

    run = true;
    step = 0;
}

double Ising::symplectic_update(CSR &csr_cpu, CSR_GPU &csr_gpu, std::ofstream& file, int max_steps)
{
    reset_initialization(csr_cpu, csr_gpu);
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(1, 1);
    blocksPerGrid.x = ceil(double(ncols) / double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(nrows) / double(threadsPerBlock.y));

    long int total_time = 0;

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    int j_max = 1;
    int i_max = sampling_period;

    for (int j = 0; j < j_max; j++) {
        for (int i = 0; i < i_max; i++) {
        symplectic_kernel_maxcut<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(csr_gpu.d_row_indices, csr_gpu.d_columns, csr_gpu.d_data, X_gpu, Y_gpu, pressure_slope, time_step, step_gpu, xi0, ncols, nrows);
    }

    update_window<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(X_gpu, current_spins_gpu, final_spins_gpu, bifurcated_gpu,
                                                            prev_bifurcated_gpu, stability_gpu, convergence_threshold, ncols, nrows);
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    auto start4 = std::chrono::high_resolution_clock::now();

    pressure = pressure_slope * time_step * step;
    pressure = pressure < 1.0 ? pressure : 1.0;

    Update_XY<<<blocksPerGrid, threadsPerBlock>>>(X_gpu, Y_gpu, pressure, time_step, ncols, nrows);

    while (run)
    {
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);

        step += j_max * i_max;
        run = (step < max_steps);
    }

    auto stop4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(stop4 - start4);
    total_time += duration4.count();

    compute_max_cut<<<blocksPerGrid, threadsPerBlock>>>(csr_gpu.d_row_indices, csr_gpu.d_columns, csr_gpu.d_data,
                                                        current_spins_gpu, cut_array_gpu, ncols, nrows);

    sum_max_cut<<<blocksPerGrid, threadsPerBlock>>>(cut_array_gpu, final_cut_array_gpu, csr_cpu.sum, ncols, nrows);
    cudaMemcpy(final_cut_array_cpu, final_cut_array_gpu, ncols * sizeof(double), cudaMemcpyDeviceToHost);

    double max_cut = -DBL_MAX;

    for (int i = 0; i < ncols; i++)
    {
        if (final_cut_array_cpu[i] > max_cut) max_cut = final_cut_array_cpu[i];
    }

    file << total_time << ", ";


    return max_cut;
}

void Ising::generate_random(CSR &csr_cpu, CSR_GPU &csr_gpu) {
    // int ncols = agents;
    int nrows = csr_cpu.size;

    std::vector<int> spins;

    for (int i = 0; i < nrows; i++) {
        spins.push_back((std::rand() % 10 < 5) ? 1 : -1);
    }

    double sum = 0.0f;

    for (int i = 0; i < nrows; i++) {
        int start_pointer = csr_cpu.csr_row_indices.at(i);
        int end_pointer = csr_cpu.csr_row_indices.at(i+1);
        for (int j = start_pointer; j < end_pointer; j++) {
            int col = csr_cpu.csr_columns.at(j);
            if ( spins.at(i) != spins.at(col)) sum += csr_cpu.csr_data.at(j);
        }
    }
    std::cout << "\n";

    std::cout << "max cut from random spin assignment: " << sum * 0.5 << "\n";
    
}