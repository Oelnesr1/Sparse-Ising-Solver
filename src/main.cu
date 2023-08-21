#include "csr.cuh"
#include "kernels.cuh"

void max_cut(string filename) {
    CSR csr_cpu(filename);
    std::ofstream file("dump.csv");
    std::cout << "created csr cpu!\n";
    CSR_GPU csr_gpu(csr_cpu);
    std::cout << "created csr gpu!\n";

    Ising ising(csr_cpu, csr_gpu);
    std::cout << "created ising model!\n";

    ising.symplectic_update(csr_cpu, csr_gpu, file);
    std::cout << "completed symplectic update!\n";

    ising.generate_random(csr_cpu, csr_gpu);
    std::cout << "completed random spin assignment generation!\n";
}

void simulate_tsp(string filename) {
    TSP_CPU tsp_cpu(filename);
    std::ofstream file("dump.csv");
    std::cout << "created tsp cpu!\n";
    TSP_GPU tsp_gpu(tsp_cpu);
    std::cout << "created tsp gpu!\n";

    vector<double> full_matrix = tsp_gpu.construct_matrix();
    CSR csr_cpu(full_matrix, pow(tsp_gpu.ncities,2));
    std::cout << "created csr cpu!\n";
    CSR_GPU csr_gpu(csr_cpu);
    std::cout << "created csr gpu!\n";

    Ising_TSP ising_tsp(tsp_gpu, csr_cpu, csr_gpu);
    std::cout << "created ising model!\n";

    ising_tsp.symplectic_update(tsp_gpu, csr_cpu, csr_gpu, file, 10'000);
    std::cout << "completed symplectic update!\n";

    ising_tsp.generate_random(tsp_cpu, tsp_gpu);
    std::cout << "completed random spin assignment generation!\n";
}

void test_max_cut() {
    std::ifstream fin("/datadrive/axelf/get-graphs/data/file_list.txt");
    std::ofstream file("simulated_bifurcation_other_timing_1_v4.csv.csv");

    vector<double> step_vector{50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    // vector<double> step_vector{200, 1000, 2000, 50000, 10000};
    file << "total time, file name, nrows, nonzeros, nsteps, max cut\n";
    string file_name;

    for (int i = 0; i < 40; i++) {
        fin >> file_name;
        string full_name = "/datadrive/axelf/get-graphs/data/" + file_name;
        std::cout << i << ", " << file_name << "\n";
        CSR csr_cpu_loop(full_name);
        CSR_GPU csr_gpu_loop(csr_cpu_loop);
        Ising ising_loop(csr_cpu_loop, csr_gpu_loop);
        for (int j = 0; j < step_vector.size(); j++) {
            int max_steps = step_vector.at(j);
            double max_cut = ising_loop.symplectic_update(csr_cpu_loop, csr_gpu_loop, file, max_steps);
            file << file_name << ", " << csr_cpu_loop.size << ", " << csr_cpu_loop.nonzeros << ", "  << max_steps <<  ", " << max_cut << "\n";
        }
    }
    fin.close();
    file.close();

}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "usage: ./program <market matrix file>\n";
        return -1;
    }
    srand(std::time(0));
    // srand(2);

    // max_cut(argv[1]);

    simulate_tsp(argv[1]);

    // test_max_cut();

    return 0;
}
