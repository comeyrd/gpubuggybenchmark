#include "bilateral.hpp"
#include <cstring>
template <int R>
void Bilateral::run_cpu(const BilateralData &bData, const BilateralSettings &bSettings, BilateralResult &bResult) {
#pragma omp parallel for collapse(2)
    for (int idx = 0; idx < bData.width; idx++) {
        for (int idy = 0; idy < bData.height; idy++) {

            int id = idy * bData.width + idx;
            float I = bData.inputImage[id];
            float res = 0.f;
            float normalization = 0.f;

// window centered at the coordinate (idx, idy)
#pragma unroll
            for (int i = -R; i <= R; i++) {
#pragma unroll
                for (int j = -R; j <= R; j++) {

                    int idk = idx + i;
                    int idl = idy + j;

                    // mirror edges
                    if (idk < 0)
                        idk = -idk;
                    if (idl < 0)
                        idl = -idl;
                    if (idk > bData.width - 1)
                        idk = bData.width - 1 - i;
                    if (idl > bData.height - 1)
                        idl = bData.height - 1 - j;

                    int id_w = idl * bData.width + idk;
                    float I_w = bData.inputImage[id_w];

                    // range kernel for smoothing differences in intensities
                    float range = -(I - I_w) * (I - I_w) / (2.f * bSettings.variance_I);

                    // spatial kernel for smoothing differences in coordinates
                    float spatial = -((idk - idx) * (idk - idx) + (idl - idy) * (idl - idy)) / (2.f * bSettings.variance_spatial);

                    // combined weight
                    float weight = bSettings.a_square * expf(spatial + range);
                    normalization += weight;
                    res += (I_w * weight);
                }
            }
            bResult.outputImage[id] = res / normalization;
        }
    }
}

void Bilateral::register_cli_options(argparse::ArgumentParser &parser) {
    auto &group = parser.add_mutually_exclusive_group(true);
    group.add_argument("-a", "--all").flag();
    group.add_argument("-cv", "--choose-version").nargs(argparse::nargs_pattern::at_least_one);
    group.add_argument("-lv", "--list-versions").flag();
    parser.add_argument("--repetitions", "-r").default_value(100).scan<'i', int>().help("Number of repetitions of execution of each Version");
    parser.add_argument("--warmup", "-w").default_value(5).scan<'i', int>().help("How many executions must be used as warm up");
};

int Bilateral::run_kernel(int argc, char **argv) {
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
        std::cout << "Versions of Bilateral :" << std::endl;
        for (const auto &version : this->list_versions()) {
            std::cout << version << std::endl;
        }
    }else{
        class_umap<IBilateral> versions_map;
        if (all_set) {
            versions_map = Manager<IBilateral>::instance()->getClasses();
        } else if (versions_set) {
            std::vector<std::string> versions = fpc_parser.get<std::vector<std::string>>("-cv");
            class_umap<IBilateral> versions_map = select_versions_in_umap(versions, Manager<IBilateral>::instance()->getClasses());
        } else {
            std::cout << fpc_parser << std::endl;
            return 1;
        }
        BilateralSettings bSettings = {WIDTH,HEIGHT, A_SQUARE,VARIANCE_I,VARIANCE_SPATIALE};
        BilateralData bData = this->random_data(bSettings);
        this->run_versions(versions_map, repetitions, warmup,bData,bSettings);
    }
    return 0;
}

std::vector<std::string> Bilateral::list_versions() {
    class_umap<IBilateral> versions = Manager<IBilateral>::instance()->getClasses();
    std::vector<std::string> vs;
    for (const auto &[name, _] : versions) {
        vs.push_back(name);
    }
    return vs;
}

void Bilateral::run_versions(class_umap<IBilateral> versions, int repetitions, int warmup, BilateralData bData, BilateralSettings bSettings) {

    BilateralResult baseResult;
    baseResult.size = bSettings.height * bSettings.width;
    baseResult.b_size = baseResult.size * sizeof(float);
    baseResult.outputImage = (float *)malloc(baseResult.b_size);
    this->run_cpu<4>(bData,bSettings,baseResult);
    baseResult.updateChecksum();
    BilateralResult vResult;
    vResult.size = bSettings.height * bSettings.width;
    vResult.b_size = vResult.size * sizeof(float);
    vResult.outputImage = (float *)malloc(vResult.b_size);
    for (const auto &[name, version_impl] : versions) {
        std::cout << "Version " << name << std::endl;
        try {
            KernelStats vStat = this->run_impl(version_impl, repetitions, warmup, bData, bSettings, vResult);
            vResult.updateChecksum();
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
    free(bData.inputImage);
    free(baseResult.outputImage);
    free(vResult.outputImage);
}

KernelStats Bilateral::run_impl(std::shared_ptr<IBilateral> bilateral_impl, int repetitions, int warmup, BilateralData &bData, BilateralSettings &bSettings, BilateralResult &bResult) {
    KernelStats kstats;
    StableMeanCriterion criterion(1000000, 5, 1);
    for (int _ = 0; _ < warmup; _++) {
        bilateral_impl->bilateral(bData, bSettings, bResult);
    }
    while (!criterion.should_stop()) {
        KernelStats kstats = bilateral_impl->bilateral(bData, bSettings, bResult);
        criterion.observe(kstats);
    }
    reset_gpu();

    return criterion.get_mean();
}

BilateralData Bilateral::random_data(const BilateralSettings &bSettings) {
    BilateralData bData;
    bData.width = bSettings.width;
    bData.height = bSettings.height;
    bData.size = bSettings.width * bSettings.height;
    bData.b_size = bData.size * sizeof(float);
    bData.inputImage = (float *)malloc(bData.b_size);

    srand(123);
    for (int i = 0; i < bData.size; i++)
        bData.inputImage[i] = rand() % 256;

    return bData;
}

REGISTER_CLASS(IKernel, Bilateral);

uint32_t BilateralResult::computeChecksum() const {
    {
        if (!outputImage || size == 0)
            return 0;
        uint32_t sum = 0;
        for (size_t i = 0; i < size; ++i) {
            uint32_t val;
            memcpy(&val, &outputImage[i], sizeof(uint32_t));
            sum ^= val;
        }
        return sum;
    }
}