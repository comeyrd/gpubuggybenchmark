#ifndef Kernel_stats_hpp
#define Kernel_stats_hpp
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>      // std::string
#include "Types.hpp"

#include <iostream>
#include <chrono>

template <typename Func>
void MeasureCpuTime(const std::string& name, Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << name << " took " << duration.count() << " ms" << std::endl;
}

struct KernelStats{
    int m_warmup;
    int m_repetitions;
    float memcpy2D = 0;//Mem init + copy 2device
    float memcpy2H = 0;//copy back 2 host
    float* warmup_duration;
    float* repetitions_duration;
    float mean_warmup;
    float mean_repetitions;
    int nb_w;
    int nb_r;
    bool str_ver_ker = false;//if the kernel and version have been allocated and filled
    std::string kernel;
    std::string version;

    explicit KernelStats(const int& warmup, const int& repetitions):m_warmup(warmup), m_repetitions(repetitions){
        warmup_duration = new float[m_warmup];
        repetitions_duration = new float[m_repetitions];
    }
    ~KernelStats(){
        delete warmup_duration;
        delete repetitions_duration;
    }
    void set_kernel_version(const std::string& kernel_, const std::string &version_){
        kernel = kernel_;
        version = version_;
        str_ver_ker = true;
        compute_mean();
    }
    void compute_mean(){
        float total = 0;
        for(int w = 0; w < nb_w;w++){
            total+=warmup_duration[w];
        }
        mean_warmup = total / nb_w;
        total = 0;
        for(int r = 0; r < nb_r;r++){
            total+=repetitions_duration[r];
        }
        mean_repetitions = total / nb_r;
    }
    KernelStats(const KernelStats& other)
        : memcpy2D(other.memcpy2D),
          memcpy2H(other.memcpy2H),
          mean_warmup(other.mean_warmup),
          mean_repetitions(other.mean_repetitions),
          nb_w(other.nb_w),
          nb_r(other.nb_r),
          str_ver_ker(other.str_ver_ker),
          kernel(other.kernel),
          version(other.version)
    {
        // allocate only as many slots as the original object currently uses
        warmup_duration     = new float[nb_w];
        repetitions_duration = new float[nb_r];

        std::copy(other.warmup_duration,
                  other.warmup_duration + nb_w,
                  warmup_duration);
        std::copy(other.repetitions_duration,
                  other.repetitions_duration + nb_r,
                  repetitions_duration);
    }
    KernelStats& operator=(const KernelStats&) = delete;
};
    inline std::ostream& operator<<(std::ostream& os, KernelStats &e_stat) {
    if(e_stat.mean_repetitions == 0){
       e_stat.compute_mean();
    }
    os  << "Warmup mean per kernel: " << e_stat.mean_warmup <<"| repetitions mean per kernel: " << e_stat.mean_repetitions << " | memcpy H2D : "<<  e_stat.memcpy2D << "  | memcpy D2H : "<< e_stat.memcpy2H << " ms" << std::endl;
    return os;
};

template <typename T>
struct CSVExportable;

template <>
struct CSVExportable<KernelStats> {
    static std::string header() {
        return "memcpy2D,memcpy2H,warmup_duration,repetitions_duration,warmup,repetitions,kernel,version";
    }

    static std::string values(const KernelStats& ks) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << ks.memcpy2D << "," << ks.memcpy2H << "," ;
        for (int w = 0; w < ks.nb_w-1;w++){
            oss << ks.warmup_duration[w] << "|";
        }
        oss << ks.warmup_duration[ks.nb_w-1] << ",";

        for (int r = 0; r < ks.nb_r-1;r++){
            oss << ks.repetitions_duration[r] << "|";
        }
        oss << ks.repetitions_duration[ks.nb_r-1] << ",";
        
        oss << ks.m_warmup << "," << ks.m_repetitions;

        if (ks.str_ver_ker){
            oss << "," << ks.kernel << "," << ks.version;    
        }else{
            oss << ",";
        }
        return oss.str();
    }
};

template <typename Iterable>
void exportCsvToStream(const Iterable& data,std::ostream &os,bool with_header){
    using T = typename Iterable::value_type;
    static_assert(std::is_class_v<CSVExportable<T>>, 
                  "CSVExportable<T> specialization required for this type");
    if (with_header){
        os << CSVExportable<T>::header() << std::endl;
    }
    for (const auto& item : data){
        os << CSVExportable<T>::values(item) << std::endl;
    }
}

template <typename Iterable>
void exportCsv(const Iterable& data, std::string filename){
    using T = typename Iterable::value_type;
    static_assert(std::is_class_v<CSVExportable<T>>, 
                  "CSVExportable<T> specialization required for this type");
    std::ifstream infile(filename);
    bool create_new_file = true;
    bool file_exists = false;
    if (infile.is_open()){
        file_exists = true;
        std::string firstLine;
        if (std::getline(infile,firstLine)){
            if(firstLine == CSVExportable<T>::header()){
                create_new_file = false;
            }
        }
    }
    if(file_exists && create_new_file){
        std::cout << filename << " already exists, but headers doesnt match. Writing in " << filename <<".new";
        filename = filename + ".new";
    }
    std::ofstream file;
    if(create_new_file){
        file = std::ofstream(filename);
    }else{
        file = std::ofstream(filename,std::ios::app);
    }

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    exportCsvToStream(data,file,create_new_file);
}

#endif 