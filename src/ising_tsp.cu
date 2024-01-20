#include "csr.cuh"
#include "kernels.cuh"
#include "helper_cuda.h"

Ising_TSP::Ising_TSP(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu)
{
    initialize(tsp_gpu, csr_cpu, csr_gpu);
}

void Ising_TSP::initialize(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu)
{
    ncities = tsp_gpu.ncities;
    nrows = csr_cpu.size;
    ncols = agents;
    SIZE = nrows * ncols;

    double_bytes_SIZE = SIZE * sizeof(double);
    int_bytes_SIZE = SIZE * sizeof(int);
    int_bytes_nrows = nrows * sizeof(int);
    bool_bytes_ncols = ncols * sizeof(bool);

    X_cpu = (double *)malloc(double_bytes_SIZE);
    Y_cpu = (double *)malloc(double_bytes_SIZE);
    average_spin_cpu = (double*)malloc(nrows * sizeof(double));
    final_spins_cpu = (int*)malloc(int_bytes_SIZE);
    current_spins_cpu = (int*)malloc(int_bytes_SIZE);
    valid_list_cpu = (int*)malloc(ncols*sizeof(int));

    run = true;
    step = 0;

    checkCudaErrors(cudaMalloc(&X_gpu, double_bytes_SIZE));
    checkCudaErrors(cudaMalloc(&Y_gpu, double_bytes_SIZE));

    checkCudaErrors(cudaMalloc(&current_spins_gpu, int_bytes_SIZE));
    checkCudaErrors(cudaMalloc(&final_spins_gpu, int_bytes_SIZE));
    checkCudaErrors(cudaMalloc(&average_spin_gpu, nrows * sizeof(double)));

    checkCudaErrors(cudaMalloc(&stability_gpu, int_bytes_SIZE));
    checkCudaErrors(cudaMalloc(&bifurcated_gpu, bool_bytes_ncols));
    checkCudaErrors(cudaMalloc(&prev_bifurcated_gpu, bool_bytes_ncols));

    checkCudaErrors(cudaMalloc(&valid_list_gpu, ncols * sizeof(int)));
    checkCudaErrors(cudaMalloc(&city_visits_gpu, (ncols * ncities) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&order_visits_gpu, (ncols * ncities) * sizeof(int)));

    checkCudaErrors(cudaMalloc(&step_gpu, sizeof(int)));
}

void Ising_TSP::reset_initialization(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu) {
    nrows = csr_cpu.size;
    ncols = agents;
    SIZE = nrows * ncols;

    for (int i = 0; i < SIZE; i++)
    {
        X_cpu[i] = 0;
        Y_cpu[i] = (((double)rand() / RAND_MAX) * 2 - 1)/10;
    }

    if (optimal_xi0) xi0 = 0.002;
    else {
        double sum = 0.0f;
        double nonzero_contribution = 0.0f;
        double zero_contribution = 0.0f;

        for (int i = 0; i < csr_cpu.csr_data.size(); i++)
        {
            sum += csr_cpu.csr_data[i];
        }

        double mean = sum / (nrows*nrows);

        for( int i = 0; i < csr_cpu.nonzeros; i++) {
            // nonzero_contribution += std::pow((csr_cpu.csr_data[i] - mean), 2);
            nonzero_contribution += std::pow(csr_cpu.csr_data[i], 2);
        }

        // zero_contribution = (nrows*nrows - csr_cpu.nonzeros) * mean * mean;

        sum = nonzero_contribution + zero_contribution;

        double variance = std::pow((sum / (nrows*nrows - 1)), 0.5);
        xi0 = 0.5 / (variance * std::pow(nrows, 0.5));
    }

    A = tsp_gpu.A;
    B = tsp_gpu.B;
    C = tsp_gpu.C;

    checkCudaErrors(cudaMemcpy(X_gpu, X_cpu, double_bytes_SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Y_gpu, Y_cpu, double_bytes_SIZE, cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMemset(current_spins_gpu, -1, int_bytes_SIZE));
    checkCudaErrors(cudaMemset(final_spins_gpu, -1, int_bytes_SIZE));

    checkCudaErrors(cudaMemset(bifurcated_gpu, 0, bool_bytes_ncols));
    checkCudaErrors(cudaMemset(prev_bifurcated_gpu, 0, bool_bytes_ncols));
    checkCudaErrors(cudaMemset(valid_list_gpu, 0, ncols*sizeof(int)));

    checkCudaErrors(cudaMemset(city_visits_gpu, 0, ncities*ncols*sizeof(int)));
    checkCudaErrors(cudaMemset(order_visits_gpu, 0, ncities*ncols*sizeof(int)));
    checkCudaErrors(cudaMemset(step_gpu, 0, sizeof(int)));

    run = true;
    step = 0;

}

double Ising_TSP::symplectic_update(TSP_GPU &tsp_gpu, CSR& csr_cpu, CSR_GPU& csr_gpu, std::ofstream& file, int max_steps)
{
    reset_initialization(tsp_gpu, csr_cpu, csr_gpu);
    int ncols = agents;
    int nrows = csr_cpu.size;
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(1, 1);
    blocksPerGrid.x = ceil(double(ncols) / double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(nrows) / double(threadsPerBlock.y));

    long int total_time = 0;

    // prepare the CUDA graph

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    int j_max = 1;
    sampling_period = max_steps / 10;
    int i_max = sampling_period;
    int n = 2;

    for (int j = 0; j < j_max; j++) {
        for (int i = 0; i < i_max; i++) {
            symplectic_kernel_tsp<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(csr_gpu.d_row_indices, csr_gpu.d_columns, csr_gpu.d_data,
             tsp_gpu.total_distances_gpu, X_gpu, Y_gpu, time_step, step_gpu, max_steps, xi0, n, A, B, C, ncities, ncols, nrows);
    }

    update_window_tsp<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(X_gpu, current_spins_gpu, final_spins_gpu, bifurcated_gpu,
    prev_bifurcated_gpu, stability_gpu, valid_list_gpu, convergence_threshold, city_visits_gpu, order_visits_gpu, step_gpu, ncities, ncols, nrows);
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // double pressure = 0;

    // start timing the algorithm

    auto start4 = std::chrono::high_resolution_clock::now();

    Update_XY<<<blocksPerGrid, threadsPerBlock>>>(X_gpu, Y_gpu, step_gpu, max_steps, time_step, ncols, nrows);

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

    // finish timing the algorithm, copy everything important from the GPU to the CPU

    tsp_spin_average<<<blocksPerGrid, threadsPerBlock>>>(current_spins_gpu, average_spin_gpu, ncities, ncols, nrows);
    cudaMemcpy(average_spin_cpu, average_spin_gpu, nrows*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(final_spins_cpu, final_spins_gpu, int_bytes_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(current_spins_cpu, current_spins_gpu, int_bytes_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(valid_list_cpu, valid_list_gpu, ncols*sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "\n";

    vector<int> valid_agents;
    vector<double> all_distances;
    double total_sum = 0;
    double minimum = DBL_MAX;
    double maximum = - DBL_MAX;

    // get all valid solutions and the corresponding agent

    int valid_col = -1;
    int valid_sum = 0;
    bool first_valid = 1;
    for (int i = 0; i < ncols; i++) {
        if (valid_list_cpu[i] > 0) {
            valid_sum += 1;
            valid_agents.push_back(i);
            if (first_valid) {
            valid_col = i;
            first_valid = 0;
            }
        }
    }

    for (int j = 0; j < valid_sum; j++) {

    int cur_agent = valid_agents.at(j);

    vector<int> cities;

    for (int order = 0; order < ncities; order++) {
        for (int city = 0; city < ncities; city++) {
            if (final_spins_cpu[(order*ncities + city) * ncols + cur_agent] == 1) {
                cities.push_back(city);
            }
        }
    }

    double tour_distance = calculate_distance_of_tour(tsp_gpu, cities, ncities);
    all_distances.push_back(tour_distance);

        total_sum += tour_distance;
        if (tour_distance < minimum) {
            minimum = tour_distance;
        }
        if (tour_distance > maximum) {
            maximum = tour_distance;
        }

    }

    if (valid_col == -1) {
        std::cout << "no agent had a valid solution\n";
    }

    double average = total_sum / valid_sum;
    double std_dev = 0;

    for (int i =0; i < all_distances.size(); i++) {
        std_dev += pow(all_distances.at(i) - average, 2);
    }
    std_dev = pow(std_dev / (valid_sum-1), 0.5);

    // std::cout << "\nvalid sum: " << valid_sum << "\n";
    
    // std::cout << "minimum: " << minimum << "\n";
    // std::cout << "maximum: " << maximum << "\n";
    // std::cout << "average: " << average << "\n";
    // std::cout << "standard deviation: " << std_dev << "\n";

    file << total_time << ", " << minimum << ", " << maximum << ", " << average << ", " << std_dev << ", " << valid_sum << ", ";
    std::cout << "time: " << total_time << ", vagents: " << valid_sum << ", steps: " << max_steps << ", max: " << maximum << ", min: " << minimum << ", average: " << average << ", Std Dev: " << std_dev << "\n";

    return 0;
    
}

void Ising_TSP::generate_random(TSP_CPU& tsp_cpu, TSP_GPU &tsp_gpu) {
    int ncities = tsp_cpu.ncities;
    std::vector<int> cities;

    for (int i = 0; i < ncities; i++) {
        cities.push_back(i);
    }
    std::random_shuffle(std::begin(cities), std::end(cities));

    double tour_distance = calculate_distance_of_tour(tsp_gpu, cities, ncities);
    std::cout << "\n";

    std::cout << "traveling salesman distance from random spin assignment: " << tour_distance << "\n";

    
}

double Ising_TSP::calculate_distance_of_tour(TSP_GPU &tsp_gpu, vector<int> cities, int ncities) {
    double sum = 0.0f;
    int city1, city2;
    for (int i = 0; i < ncities; i++) {
        if (i == ncities-1) {
            city1 = min(cities.at(ncities-1), cities.at(0));
            city2 = max(cities.at(ncities-1), cities.at(0));
        } else {
            city1 = min(cities.at(i), cities.at(i+1));
            city2 = max(cities.at(i), cities.at(i+1));
        }
        int pos = city2 + (city1 * ncities) - ((city1+2)*(city1+1)/ 2);
        sum += tsp_gpu.cpu_distances[pos];
    }
    return sum;
}