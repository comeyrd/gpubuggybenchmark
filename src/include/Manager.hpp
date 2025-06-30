#ifndef MANAGER_HPP
#define MANAGER_HPP

//TODO Create an Interface that could be implemented by the "kernel manager" for each type of kernel

template <class _Tp> using kernel_umap = std::unordered_map<std::string, std::shared_ptr<_Tp>>;

template<class _Tp> class Manager{
    private:
         kernel_umap<_Tp> _kernels;
    public:
    
        static Manager* instance(){
            static Manager manager;
            return &manager;
        }
        const kernel_umap<_Tp> &getKernels(){
            return _kernels;
        };

        void registerKernel(const std::string& name, std::shared_ptr<_Tp> kernel){
            _kernels[name] = kernel;
        };
        
};

#define REGISTER_CLASS(InterfaceName,ClassName) \
    namespace { \
        struct ClassName##AutoRegister { \
            ClassName##AutoRegister() { \
                Manager<InterfaceName>::instance()->registerKernel(#ClassName, std::make_shared<ClassName>()); \
            } \
        }; \
        static ClassName##AutoRegister global_##ClassName##AutoRegister; \
}
#endif