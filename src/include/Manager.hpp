#ifndef MANAGER_HPP
#define MANAGER_HPP
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

template <class _Tp> using class_umap = std::unordered_map<std::string, std::shared_ptr<_Tp>>;

template<class _Tp> using class_pair = std::pair<const std::string,std::shared_ptr<_Tp>>;

template <class _Tp > class_umap<_Tp> select_versions_in_umap(const std::vector<std::string>& keys, const class_umap<_Tp>& input_map){
    class_umap<_Tp> result;
    for (const auto &key : keys){
        if (input_map.find(key) != input_map.end()) {
            result[key] = input_map.at(key);
        }else{
            std::cerr << "Version " << key << " not found" << std::endl;
        }
    }
    return result;
}

template<class _Tp> class Manager{
    private:
         class_umap<_Tp> _classes;
    public:
    
        static Manager* instance(){
            static Manager manager;
            return &manager;
        }
        const class_umap<_Tp> &getClasses(){
            return _classes;
        };

        void register_class(const std::string& name, std::shared_ptr<_Tp> impl){
            _classes[name] = impl;
        };
        
};

template<class _Int, class _Impl> class Registrar{
    public:
    Registrar(const std::string& name){
        Manager<_Int>::instance()->register_class(name,std::make_shared<_Impl>());
    }
};

#define REGISTER_CLASS(Interface, Class) \
    static Registrar<Interface, Class> _registrar_##Class(#Class);

#endif