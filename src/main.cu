#include "csr.cuh"
#include "kernels.cuh"

void simulate_tsp(string filename)
{
    TSP_CPU tsp_cpu(filename);
    std::ofstream file("dump.csv");
    std::cout << "created tsp cpu!\n";
    TSP_GPU tsp_gpu(tsp_cpu);
    std::cout << "created tsp gpu!\n";

    vector<double> full_matrix = tsp_gpu.construct_matrix();
    CSR csr_cpu(full_matrix, pow(tsp_gpu.ncities, 2));
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

void test_max_cut(bool gset, bool uw, string ofile, string ifile = "")
{

    ifile = "/datadrive/axelf/get-graphs/data/file_list.txt";
    // ofile = "simulated_bifurcation_other_timing_1_v4.csv.csv";

    if (gset)
    {
        std::ofstream file(ofile);
        vector<int> step_vector{0, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000};
        // vector<int> step_vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        // vector<int> step_vector{500};

        // 10000, 10000, 10000, 10000, 10000, 25000, 25000, 25000, 25000, 25000, 50000, 50000, 50000, 50000, 50000};
        // vector<int> graphs_to_attempt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, };
        std::ifstream metadata_file("../MAXCUT2/metadata.txt");
        // vector<double> step_vector{200, 1000, 2000, 50000, 10000};
        file << "time,minimum,maximum,average,std dev,ps,file name,nrows,nonzeros,nsteps\n";

        for (int i = 1; i < 42; i++)
        {
            // int index = graphs_to_attempt.at(i);
            bool run = true;
            string full_name = "/home/marhamil/axelf/primes/MAXCUT2/G" + std::to_string(i) + ".mtx";
            string file_name = "G" + std::to_string(i);
            double best;
            string dump;
            while (dump != file_name)
            {
                metadata_file >> dump >> best;
                std::cout << dump << ", " << best << "\n";
            }
            // std::cout << index << ", " << file_name << "\n";
            CSR csr_cpu_loop(full_name);
            CSR_GPU csr_gpu_loop(csr_cpu_loop);
            Ising ising_loop(csr_cpu_loop, csr_gpu_loop);
            int k = 0;
            while (run)
            {
                for (int j = 0; j < step_vector.size(); j++)
                {
                    int max_steps = step_vector.at(j);
                    double max_cut = ising_loop.symplectic_update(csr_cpu_loop, csr_gpu_loop, file, uw, max_steps);
                    file << file_name << ", " << csr_cpu_loop.size << ", " << csr_cpu_loop.nonzeros << ", " << max_steps << "\n";
                    if (max_cut == best)
                        run = false;
                }
                k++;
                if (k == 5)
                    run = false;
            }
        }
        file.close();
    }
    else
    {
        std::ifstream fin(ifile);
        std::ofstream file(ofile);
        vector<double> step_vector{50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
        // vector<double> step_vector{200, 1000, 2000, 50000, 10000};
        file << "time,minimum,maximum,average,std dev,ps,file name,nrows,nonzeros,nsteps\n";
        string file_name;

        for (int i = 0; i < 37; i++)
        {
            fin >> file_name;
            string full_name = "/datadrive/axelf/get-graphs/data/" + file_name;
            std::cout << i << ", " << file_name << "\n";
            CSR csr_cpu_loop(full_name);
            CSR_GPU csr_gpu_loop(csr_cpu_loop);
            Ising ising_loop(csr_cpu_loop, csr_gpu_loop);
            for (int j = 0; j < step_vector.size(); j++)
            {
                int max_steps = step_vector.at(j);
                double max_cut = ising_loop.symplectic_update(csr_cpu_loop, csr_gpu_loop, file, uw, max_steps);
                file << file_name << ", " << csr_cpu_loop.size << ", " << csr_cpu_loop.nonzeros << ", " << max_steps << "\n";
            }
        }
        fin.close();
        file.close();
    }
}

void test_tsp(bool uw, string ofile)
{
    std::ofstream file(ofile);
    vector<double> step_vector{50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    vector<string> file_names{"burma14.tsp", "dj38.tsp", "ulysses16.tsp", "ulysses22.tsp", "wi29.tsp"};
    // vector<double> step_vector{200, 1000, 2000, 50000, 10000};
    file << "total time, minimum, maximum, average, std dev, valid agents, file name, ncities, nsteps\n";
    string file_name;

    for (int i = 0; i < file_names.size(); i++)
    {
        file_name = file_names.at(i);
        // string full_name = "/datadrive/axelf/get-graphs/data/" + file_name;
        std::cout << i << ", " << file_name << "\n";
        TSP_CPU tsp_cpu(file_name);
        TSP_GPU tsp_gpu(tsp_cpu);
        vector<double> full_matrix = tsp_gpu.construct_matrix();
        CSR csr_cpu(full_matrix, pow(tsp_gpu.ncities, 2));
        CSR_GPU csr_gpu(csr_cpu);
        Ising_TSP ising_tsp(tsp_gpu, csr_cpu, csr_gpu);

        for (int j = 0; j < step_vector.size(); j++)
        {
            int max_steps = step_vector.at(j);
            double max_cut = ising_tsp.symplectic_update(tsp_gpu, csr_cpu, csr_gpu, file, max_steps);
            file << file_name << ", " << tsp_gpu.ncities << ", " << csr_cpu.nonzeros << ", " << max_steps << "\n";
        }
    }
    file.close();
}

void test_fused_kernel_maxcut(bool uw, bool fused, string ofile, string ifile = "")
{
    if (ifile.empty()) ifile = "/datadrive/axelf/get-graphs/data/file_list.txt";

    // ofile = "simulated_bifurcation_other_timing_1_v4.csv.csv";

    std::ofstream file(ofile);
    // vector<int> agent_vector{1, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000};
    vector<int> agent_vector{1000, 750, 500, 250, 100, 75, 50, 25, 10, 1};
    // vector<double> time_step_vector{1.25, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75, 2};
    vector<double> time_step_vector{2, 1.75, 1.5, 1.25, 1.2, 1.15, 1.1, 1.05, 1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.5, 0.25, 0.1, 0.05, 0.1};
    std::ifstream metadata_file("../MAXCUT2/metadata.txt");
    file << "time,minimum,maximum,average,std dev,ps,file name,nrows,nonzeros,nsteps,time step,nagents\n";

    int total_time = 10'000;

    for (int i = 1; i < 68; i++)
    {
        string full_name = "/home/marhamil/axelf/primes/MAXCUT2/G" + std::to_string(i) + ".mtx";
        string file_name = "G" + std::to_string(i);
        std::cout << i << ", " << file_name << "\n";

        CSR csr_cpu_loop(full_name);
        CSR_GPU csr_gpu_loop(csr_cpu_loop);
        for (int a = 0; a < time_step_vector.size(); a++)
        {
            double time_step = time_step_vector.at(a);
            int temp = ceil((double)total_time / time_step);
            int max_steps = temp;
            if (temp % 25 != 0)
                max_steps += (25 - (temp %25));
            // max_steps = max_steps + (max_steps % 25);
            for (int b = 0; b < agent_vector.size(); b++)
            {
                int agents = agent_vector.at(b);
                Ising ising_loop(csr_cpu_loop, csr_gpu_loop, time_step, agents);
                double max_cut = ising_loop.symplectic_update(csr_cpu_loop, csr_gpu_loop, file, uw, fused, max_steps);
                file << file_name << "," << csr_cpu_loop.size << "," << csr_cpu_loop.nonzeros << "," << max_steps << "," << time_step << "," << agents << "\n";
            }
        }
    }

    std::ifstream fin(ifile);
    string file_name;

    for (int i = 0; i < 123; i++)
    {
        fin >> file_name;
        string full_name = "/datadrive/axelf/get-graphs/data/" + file_name;
        std::cout << i << ", " << file_name << "\n";

        CSR csr_cpu_loop(full_name);
        CSR_GPU csr_gpu_loop(csr_cpu_loop);
        for (int a = 0; a < time_step_vector.size(); a++)
        {
            double time_step = time_step_vector.at(a);
            int max_steps = ceil((double)total_time / time_step);
            max_steps = max_steps + (max_steps % 25);
            for (int b = 0; b < agent_vector.size(); b++)
            {
                int agents = agent_vector.at(b);
                Ising ising_loop(csr_cpu_loop, csr_gpu_loop, time_step, agents);
                ising_loop.symplectic_update(csr_cpu_loop, csr_gpu_loop, file, uw, fused, max_steps);
                file << file_name << "," << csr_cpu_loop.size << "," << csr_cpu_loop.nonzeros << "," << max_steps << "," << time_step << "," << agents << "\n";
            }
        }
    }

    fin.close();
    file.close();
}

void max_cut(bool uw, bool fused, double time_step, int max_steps, int agents, string ifile, string ofile)
{
    std::ofstream file(ofile, std::ios::app);
    CSR csr_cpu(ifile);
    CSR_GPU csr_gpu(csr_cpu);

    Ising ising(csr_cpu, csr_gpu, time_step, agents);
    std::cout << "file: " << ifile << ", ";
    double max_cut = ising.symplectic_update(csr_cpu, csr_gpu, file, uw, fused, max_steps);
    file << ifile << "," << csr_cpu.size << "," << csr_cpu.nonzeros << "," << max_steps << "," << time_step << "," << agents << "\n";

    file.close();

}

void traveling_salesman(bool uw, bool fused, double time_step, int max_steps, int agents, string ifile, string ofile)
{
    std::ofstream file(ofile, std::ios::app);
    TSP_CPU tsp_cpu(ifile);
    TSP_GPU tsp_gpu(tsp_cpu);
    vector<double> full_matrix = tsp_gpu.construct_matrix();
    CSR csr_cpu(full_matrix, pow(tsp_gpu.ncities, 2));
    CSR_GPU csr_gpu(csr_cpu);
    Ising_TSP ising_tsp(tsp_gpu, csr_cpu, csr_gpu);
    ising_tsp.symplectic_update(tsp_gpu, csr_cpu, csr_gpu, file, 10'000);
    ising_tsp.generate_random(tsp_cpu, tsp_gpu);
}

void get_sparsity(string ifile, string ofile) {
    CSR csr_cpu(ifile);
    std::ofstream file(ofile, std::ios::app);
    double sparsity = double(100 * csr_cpu.nonzeros) / double(std::pow(csr_cpu.size, 2));
    file << ifile << "," << sparsity << "%\n";
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "usage: ./program <default?> <.mtx/.tsp> <output file>\n";
        // std::cout << argv << "\n";
        return -1;
    }
    srand(std::time(0));

    bool mtx = false;
    bool tsp = false;
    std::string input_file;
    std::string output_file;
    bool update_window = false;
    bool fused = true;
    bool random = false;
    double time_step = 0.1;
    int max_steps = 10'000;
    int agents = 1024;


    bool default_settings = bool(std::stoi(argv[1]));
    input_file = argv[2];
    output_file = argv[3];
    if (default_settings) {
        if (ends_with(input_file, ".mtx")) {
            mtx = true;
        } else if (ends_with(input_file, ".tsp")) {
            tsp = true;
        }
    }

    else {
        if (argc != 11) {
            std::cout << "usage: ./program <default?> <.mtx/.tsp> <output file> <maxcut/tsp> <update window?> <fused?> <random?> <time_step> <max_steps> <agents>\n";
            return -1;
        }
        std::string algorithm = argv[4];
        if (algorithm == "maxcut") {
            mtx = true;
        } else if (algorithm == "tsp") {
            tsp = true;
        }

        update_window = bool(std::stoi(argv[5]));
        fused = bool(std::stoi(argv[6]));
        random = bool(std::stoi(argv[7]));
        time_step = std::stoi(argv[8]);
        max_steps = std::stoi(argv[9]);
        agents = std::stoi(argv[10]); 
    }

    if (mtx) {
        max_cut(update_window, fused, time_step, max_steps, agents, input_file, output_file);
    } else if (tsp) {
        traveling_salesman(update_window, fused, time_step, max_steps, agents, input_file, output_file);
    }





    // if automatic, determine whether we are doing maxcut or traveling salesman problem based on whether the input file is .mtx or .tsp
    // default settings: 
    // fused kernel is true, determines random based on matrix market file (if maxcut), update window is false, time_step is 0.01, max_steps is 10,000, agents is 1024




    // std::cout << argv[1] << " " << argv[2];

    // get_sparsity(argv[1], argv[2]);

    // srand(2);
    // std::cout << argv << "\n";
    // std::cout << "uw: " << bool(std::stoi(argv[1])) << ", fusion: " << bool(std::stoi(argv[2])) << ", ";

    // max_cut(bool(std::stoi(argv[1])), bool(std::stoi(argv[2])), std::stod(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), argv[6], argv[7]);

    // simulate_tsp(argv[1]);

    // test_max_cut(true, true, "gset_v6_shared_memory.csv");

    // test_fused_kernel_maxcut(false, true, "fused_kernel_test.csv");
    // test_fused_kernel_maxcut(false, false, "unfused_kernel_test.csv");

    // test_tsp(true, "tsp_v4.csv");

    return 0;
}
