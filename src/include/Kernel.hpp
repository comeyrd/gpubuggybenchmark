#ifndef Kernel_HPP
#define Kernel_HPP
#include <argparse/argparse.hpp>
#include "Manager.hpp"
#include <deque>

class IKernel{
    public:
        virtual ~IKernel() = default;
        virtual int run_kernel(int argc,char** argv) = 0;
    private: 
        virtual void register_cli_options(argparse::ArgumentParser& parser) = 0;
        virtual std::vector<std::string> list_versions() = 0;
};

typedef class_pair<IKernel> kernel_pair ;


struct KernelStats{
    float memcpy2D = 0;//Mem init + copy 2device
    float memcpy2H = 0;//copy back 2 host
    float compute = 0;//Kernel launch time

KernelStats operator+(const KernelStats& other) const {
    return {
        this->memcpy2D + other.memcpy2D,
        this->memcpy2H + other.memcpy2H,
        this->compute + other.compute,
    };
}
KernelStats operator-(const KernelStats& other) const {
    return {
        this->memcpy2D - other.memcpy2D,
        this->memcpy2H - other.memcpy2H,
        this->compute - other.compute,
    };
}
KernelStats& operator-=(const KernelStats& other) {
    this->memcpy2D -= other.memcpy2D;
    this->memcpy2H -= other.memcpy2H;
    this->compute -= other.compute;
    return *this;
}
KernelStats& operator+=(const KernelStats& other) {
    this->memcpy2D += other.memcpy2D;
    this->memcpy2H += other.memcpy2H;
    this->compute += other.compute;
    return *this;
}
KernelStats& operator/=(float scalar) {
    this->memcpy2D /= scalar;
    this->memcpy2H /= scalar;
    this->compute /= scalar;
    return *this;
}
KernelStats operator/(const KernelStats& other) const{
    return {
        this->memcpy2D / other.memcpy2D,
        this->memcpy2H / other.memcpy2H,
        this->compute / other.compute,
    };
};

KernelStats operator/(float scalar) const {
    return {
        this->memcpy2D / scalar,
        this->memcpy2H / scalar,
        this->compute / scalar,
    };
}
 bool operator<=(uint value) const {
        return std::abs(memcpy2D) <= value && std::abs(memcpy2H) <=value && std::abs(compute) <= value;
    }
};

inline std::ostream& operator<<(std::ostream& os,KernelStats e_stat) {
    os << "Memory load time, to device = " << e_stat.memcpy2D << " ms" << " to Host : "<<  e_stat.memcpy2H << " ms | Compute time =  "<< e_stat.compute << " ms";
    return os;
}

class ICriterion{
    public : 
        virtual void observe(const KernelStats& stat) = 0;
        virtual bool should_stop() const = 0;
        virtual ~ICriterion() = default;
        virtual KernelStats get_mean() const = 0;
};

class StableMeanCriterion : public ICriterion{
    public:
    StableMeanCriterion(int max_iter, int window = 5, uint percent_thresh = 5)
        : max_iteration(max_iter), window_size(window), percent_threshold(percent_thresh) {}

    void observe(const KernelStats& stat) override {
        if (recent_stats.size() == window_size)
            recent_stats.pop_front();

        recent_stats.push_back(stat);
        cumsum+=stat;
        count++;
    }
    KernelStats get_mean()const override{
        return cumsum / count;
    }

    bool should_stop() const override{
        if (count >= max_iteration){
            return true;
        }
        if (recent_stats.size() < window_size){
            return false;
        }
        return is_stable();
    }

    private:
        std::deque<KernelStats> recent_stats;
        KernelStats cumsum;
        int count = 0;
        int max_iteration;
        int window_size;
        uint percent_threshold;
        
        bool is_stable() const{
            KernelStats before_sum = cumsum;
            for (const auto &s : recent_stats){
                before_sum-=s;
            }
            int n_before = count - recent_stats.size();
            if(n_before == 0){
                return false;
            }
            KernelStats beforeMean = before_sum / n_before;
            KernelStats change = (beforeMean - (cumsum/count)) / beforeMean;

            return change <= percent_threshold;
        }

};


#endif
