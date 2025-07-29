#include "accuracy.hpp"
#include "gpu-utils.hpp"
#include <random>

int Accuracy::run_cpu(const AccuracyData &aData) {
    int count = 0;
    for (int row = 0; row < aData.n_rows; row++) {
        const int label = aData.label[row];
        const float label_pred = aData.data[row * aData.ndims + label];
        int ngt = 0;
        for (int col = 0; col < aData.ndims; col++) {
            const float pred = aData.data[row * aData.ndims + col];
            if (pred > label_pred || (pred == label_pred && col <= label)) {
                ++ngt;
            }
        }
        if (ngt <= aData.topk) {
            ++count;
        }
    }
    return count;
}

void Accuracy::register_cli_options(argparse::ArgumentParser &parser) {
    auto &group = parser.add_mutually_exclusive_group(true);
    group.add_argument("-a", "--all").flag();
    group.add_argument("-cv", "--choose-version").nargs(argparse::nargs_pattern::at_least_one);
    group.add_argument("-lv", "--list-versions").flag();
    parser.add_argument("--repetitions", "-r").default_value(100).scan<'i', int>().help("Number of repetitions of execution of each Version");
    parser.add_argument("--warmup", "-w").default_value(5).scan<'i', int>().help("How many executions must be used as warm up");
};

int Accuracy::run_kernel(int argc, char **argv) {
    std::string name;
    if (argc < 2) {
        std::cerr << "Not enough arguments" << std::endl;
        name = "-----";
    } else {
        name = std::string(argv[0]) + " " + argv[1];
    }
    argparse::ArgumentParser fpc_parser(name, VERSION_STRING, argparse::default_arguments::none);
    this->register_cli_options(fpc_parser);

    try {
        fpc_parser.parse_known_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << fpc_parser;
        return 1;
    }

    bool all_set = fpc_parser.get<bool>("--all");
    bool list_set = fpc_parser.get<bool>("--list-versions");
    int repetitions = fpc_parser.get<int>("--repetitions");
    int warmup = fpc_parser.get<int>("--warmup");
    bool versions_set = fpc_parser.is_used("-cv");

    if (list_set) {
        std::cout << "Versions of Accuracy :" << std::endl;
        for (const auto &version : this->list_versions()) {
            std::cout << version << std::endl;
        }
    } else if (all_set) {
        class_umap<IAccuracy> versions_map = Manager<IAccuracy>::instance()->getClasses();
        this->run_versions(versions_map, repetitions,warmup);
    } else if (versions_set) {
        std::vector<std::string> versions = fpc_parser.get<std::vector<std::string>>("-cv");
        class_umap<IAccuracy> versions_map = select_versions_in_umap(versions, Manager<IAccuracy>::instance()->getClasses());
        this->run_versions(versions_map, repetitions, warmup);
    } else {
        std::cout << fpc_parser << std::endl;
        return 1;
    }
    return 0;
}

std::vector<std::string> Accuracy::list_versions() {
    class_umap<IAccuracy> versions = Manager<IAccuracy>::instance()->getClasses();
    std::vector<std::string> vs;
    for (const auto &[name, _] : versions) {
        vs.push_back(name);
    }
    return vs;
}

void Accuracy::run_versions(class_umap<IAccuracy> versions, int repetitions, int warmup) {

    AccuracyData aData = this->random_data(NROWS, NDIMS, TOP_K);
    AccuracySettings aSettings = {GRID_SZ};
    AccuracyResult baseResult;
    baseResult.count = this->run_cpu(aData);

    for (const auto &[name, version_impl] : versions) {
        AccuracyResult vResult;
        std::cout << "Version " << name << std::endl;
        try {
            KernelStats vStat = this->run_impl(version_impl, repetitions, warmup, aData, aSettings, vResult);
            if (!(vResult == baseResult)) {
                std::cout << "Failed" << std::endl;
            }
            std::cout << "" << vStat << std::endl
                      << std::endl;
        } catch (std::exception &e) {
            std::cout << "Error Encountered when running impl " << name << std::endl;
            reset_gpu();
            std::cout << e.what() << std::endl;
        }
        std::cout << "--------" << std::endl;
    }
    free(aData.data);
    free(aData.label);
}

KernelStats Accuracy::run_impl(std::shared_ptr<IAccuracy> accuracy_impl, int repetitions, int warmup, AccuracyData &aData, AccuracySettings &aSettings, AccuracyResult &aResult) {
    KernelStats kstats;
    StableMeanCriterion criterion(1000000,5,1);               
    for (int _ = 0; _< warmup ; _++){
        accuracy_impl->accuracy(aData, aSettings, aResult);
    }
    while(!criterion.should_stop()){
        KernelStats kstats = accuracy_impl->accuracy(aData, aSettings, aResult);
        criterion.observe(kstats);
    }
    reset_gpu();
    
    return criterion.get_mean();
}

AccuracyData Accuracy::random_data(int n_rows, int ndims, int top_k) {
    AccuracyData aData;
    aData.n_rows = n_rows;
    aData.ndims = ndims;
    aData.topk = top_k;

    const int data_size = n_rows * ndims;

    aData.label_sz_bytes = n_rows * sizeof(int);
    aData.data_sz_bytes = data_size * sizeof(float);

    aData.label = (int *)malloc(aData.label_sz_bytes);

    srand(123);
    for (int i = 0; i < n_rows; i++)
        aData.label[i] = rand() % ndims;

    aData.data = (float *)malloc(aData.data_sz_bytes);

    std::default_random_engine g(123);
    std::uniform_real_distribution<float> distr(0.f, 1.f);
    for (int i = 0; i < data_size; i++) {
        aData.data[i] = distr(g);
    }
    return aData;
}
REGISTER_CLASS(IKernel, Accuracy);
