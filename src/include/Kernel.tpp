template <typename Data, typename Settings, typename Result>
void IKernel<Data, Settings, Result>::run(int argc, char **argv) {
    std::string name;
    if (argc < 2) {
        std::cerr << "Not enough arguments" << std::endl;
        name = "-----";
    } else {
        name = std::string(argv[0]) + " " + argv[1];
    }
    argparse::ArgumentParser kernel_parser(name, VERSION_STRING, argparse::default_arguments::none);

    auto &group = kernel_parser.add_mutually_exclusive_group(true);
    group.add_argument("-a", "--all").flag();
    group.add_argument("-cv", "--choose-version").nargs(argparse::nargs_pattern::at_least_one);
    group.add_argument("-lv", "--list-versions").flag();
    kernel_parser.add_argument("-b", "--benchmark").flag();
    kernel_parser.add_argument("--repetitions", "-r").default_value(400).scan<'i', int>().help("Number of repetitions of execution of each Version");
    kernel_parser.add_argument("--warmup", "-w").default_value(5).scan<'i', int>().help("How many executions must be used as warm up");

    // TODO the Result thingy has a way to auto register its parameters and parse or something like that ?

    try {
        kernel_parser.parse_known_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << kernel_parser;
        return;
    }

    bool all_set = kernel_parser.get<bool>("--all");
    bool list_set = kernel_parser.get<bool>("--list-versions");
    bool benchmark = kernel_parser.get<bool>("-b");
    settings.repetitions = kernel_parser.get<int>("--repetitions");
    settings.warmup = kernel_parser.get<int>("--warmup");
    bool versions_set = kernel_parser.is_used("-cv");

    class_umap<IVersion<Data,Settings,Result>> version_map;
    if (list_set) {
        std::cout << "Versions available :" << std::endl;
        for (const auto &version : list_version()) {
            std::cout << version << std::endl;
        }
    } else if (all_set) {
        version_map = Manager<IVersion<Data, Settings, Result>>::instance()->getClasses();
    } else if (versions_set) {
        std::vector<std::string> versions = kernel_parser.get<std::vector<std::string>>("-cv");
        version_map = select_versions_in_umap(versions, Manager<IVersion<Data, Settings, Result>>::instance()->getClasses());
    }else{
        std::cout << kernel_parser << std::endl;
        return;
    }
    setup_gpu();
    if(benchmark){
        run_benchmark(version_map);
    }else{
        run_versions(version_map);
    }

    return;
}

template <typename Data, typename Settings, typename Result>
KernelStats IKernel<Data, Settings, Result>::run_impl(
    std::shared_ptr<IVersion<Data, Settings, Result>> version_impl, Result &result) {
    GpuStream stream;
    GpuEventTimer timer(settings,stream.get_stream());
    version_impl->init(data,settings,result);
    timer.begin_mem2D();
    version_impl->setup();
    timer.end_mem2D();

    for(int w = 0; w < settings.warmup ; w++){
        timer.begin_warmup();
        version_impl->run(stream.get_stream());
        timer.end_warmup();
    }

    for(int r = 0; r < settings.repetitions ; r++){
        timer.begin_repetition();
        version_impl->run(stream.get_stream());
        timer.end_repetition();
    }
    timer.begin_mem2H();
    timer.end_mem2H();
    version_impl->teardown();

    return timer.retreive();
}

template <typename Data, typename Settings, typename Result>
void IKernel<Data, Settings, Result>::run_versions(
    class_umap<IVersion<Data, Settings, Result>> versions) {
    data.generate_random();
    run_cpu();
    Result vResult = Result(settings);

    for (const auto &[name, version_impl] : versions) {
        try {
            KernelStats vStat = run_impl(version_impl, vResult);
            reset_gpu();
            vStat.set_kernel_version(this->name(), name);
            if (!(vResult == cpu_result)) {
                std::cout << " Version " << name << " Failed" << std::endl;
                std::cout << vResult << " vs " << cpu_result << std::endl;
                std::cout << std::endl;
            } else {
                //std::cout << " Version " << name << " " << vStat << std::endl;
            }
            exportCsv(std::vector<KernelStats>{vStat}, "./csv/all.csv");
        } catch (std::exception &e) {
            std::cout << "Error Encountered when running impl " << name << std::endl;
            reset_gpu();
            std::cout << e.what() << std::endl;
        }
    }
}

template <typename Data, typename Settings, typename Result>
std::vector<std::string> IKernel<Data, Settings, Result>::list_version() {

    class_umap<IVersion<Data, Settings, Result>> versions = Manager<IVersion<Data, Settings, Result>>::instance()->getClasses();
    std::vector<std::string> vs;
    for (const auto &[name, _] : versions) {
        vs.push_back(name);
    }
    return vs;
}

template <typename Data, typename Settings, typename Result>
void IKernel<Data, Settings, Result>::run_benchmark(class_umap<IVersion<Data, Settings, Result>> versions) {
    std::vector<int> warmups = {};
    std::vector<int> repetitions = {10,50,125,250,375,500,750,1000,1500};
    int reruns = 1;
    int total = reruns * (warmups.size() + repetitions.size());
    int done = 0;
    for(int _ = 0; _ < reruns ; _++){
        for(int w:warmups ){
            settings.warmup = w;
            settings.repetitions = DEF_REPETITIONS;
            run_versions(versions);
            ++done;
            std::cout << "\rProgress : " << done << " / " << total << std::flush;
        }
        std::cout << std::endl;
        for(int r : repetitions){
            settings.warmup = DEF_WARMUP;
            settings.repetitions = r;
            run_versions(versions);
            ++done;
            std::cout << "\rProgress : " << done << " / " << total << std::flush;
        }
        std::cout << std::endl;
    }
    
}


