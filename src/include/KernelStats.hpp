#ifndef Kernel_stats_hpp
#define Kernel_stats_hpp
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>      // std::string

struct KernelStats{
    float memcpy2D = 0;//Mem init + copy 2device
    float memcpy2H = 0;//copy back 2 host
    float compute = 0;//Kernel launch time
    uint repetitions_inside = 0;
    uint repetitions_outside = 0;
    bool str_ver_ker = false;//if the kernel and version have been allocated and filled
    std::string kernel;
    std::string version;
    void set_kernel_version(const std::string& kernel_, const std::string &version_){
        kernel = kernel_;
        version = version_;
        str_ver_ker = true;
    }
KernelStats operator+(const KernelStats& other) const {
    return {
        this->memcpy2D + other.memcpy2D,
        this->memcpy2H + other.memcpy2H,
        this->compute + other.compute,
        this->repetitions_inside + other.repetitions_inside,
        this->repetitions_outside + other.repetitions_outside,
    };
}
KernelStats operator-(const KernelStats& other) const {
    return {
        this->memcpy2D - other.memcpy2D,
        this->memcpy2H - other.memcpy2H,
        this->compute - other.compute,
        this->repetitions_inside - other.repetitions_inside,
        this->repetitions_outside - other.repetitions_outside,
    };
}
KernelStats& operator-=(const KernelStats& other) {
    this->memcpy2D -= other.memcpy2D;
    this->memcpy2H -= other.memcpy2H;
    this->compute -= other.compute;
    this->repetitions_inside -= other.repetitions_inside;
    this->repetitions_outside -= other.repetitions_outside;
    return *this;
}
KernelStats& operator+=(const KernelStats& other) {
    this->memcpy2D += other.memcpy2D;
    this->memcpy2H += other.memcpy2H;
    this->compute += other.compute;
    this->repetitions_inside += other.repetitions_inside;
    this->repetitions_outside += other.repetitions_outside;
    return *this;
}
 bool operator<=(uint value) const {
        return std::abs(memcpy2D) <= value && std::abs(memcpy2H) <=value && std::abs(compute) <= value;
    }
};

inline std::ostream& operator<<(std::ostream& os,KernelStats e_stat) {
    os << "Repetitions inside : " <<e_stat.repetitions_inside <<" outside :"<< e_stat.repetitions_outside <<"| kernel_time per kernel: " << e_stat.compute / e_stat.repetitions_inside << " | memcpy H2D : "<<  e_stat.memcpy2D << "  | memcpy D2H : "<< e_stat.memcpy2H << " ms";
    return os;
};

template <typename T>
struct CSVExportable;

template <>
struct CSVExportable<KernelStats> {
    static std::string header() {
        return "memcpy2D,memcpy2H,compute,repetitions_inside,repetitions_outside,kernel,version";
    }

    static std::string values(const KernelStats& ks) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << ks.memcpy2D << "," << ks.memcpy2H << "," << ks.compute << "," << ks.repetitions_inside<< "," << ks.repetitions_outside;
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