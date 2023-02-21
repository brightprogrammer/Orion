#ifndef TENSORIMPL_H_
#define TENSORIMPL_H_

#include "Tensor.hpp"

#include <iomanip>
#include <cstring>
#include <cassert>

namespace Orion{

    template <typename dt>
    Tensor<dt>::Tensor(const DimVec& dim) : m_dim(dim){
        m_nelem = 1;
        for(u64 i = 0; i < rank(); i++){
            m_nelem *= m_dim[i];
        }

        m_data = reinterpret_cast<dt*>(malloc(sizeof(dt) * m_nelem));
    }

    template <typename dt>
    Tensor<dt>::Tensor(const DimVec& dim, dt* data) : m_data(data), m_dim(dim) {
        m_nelem = 1;
        for(u64 i = 0; i < rank(); i++){
            m_nelem *= m_dim[i];
        }
    }


    template <typename dt>
    inline void Tensor<dt>::zeroes(){
        memset(m_data, 0, m_nelem);
    }

    template <typename dt>
    inline void Tensor<dt>::fill(dt x){
        for(int i = 0; i < m_nelem; i++){
            m_data[i] = x;
        }
    }

    template <typename dt>
    inline void Tensor<dt>::randomize(dt min, dt max){
        srand(static_cast<u32>(time(nullptr)));
        for(u64 i = 0; i < m_nelem; i++){
            // if(max != 0) m_data[i] = min + (static_cast<dt>(rand()) - static_cast<float>(RAND_MAX)/2)*max/static_cast<dt>(RAND_MAX);
            // else m_data[i] = min + static_cast<dt>(rand()%RAND_MAX);

            m_data[i] = (static_cast<dt>(rand())/RAND_MAX)*(max-min) + min;
        }
    }

    template <typename dt>
    inline void Tensor<dt>::printLinear() const {
        for(u64 i = 0; i < m_nelem; i++){
            std::cout << std::setw(10) << m_data[i] << " ";
        }
    }

    template <typename dt>
    inline std::ostream& operator << (std::ostream& out, const Tensor<dt>& t){
        static u64 toprank = t.rank();
        if(t.rank() == 0){
            // print scalar
            std::cout << "[" << std::setw(10) << t.m_scalar << "]";
        }else if(t.rank() == 1){
            dt* m_data = t.m_data;
            const DimVec& m_dim = t.m_dim;

            // print vector
            std::cout << "[" << std::setw(10);
            for(u64 i = 0; i < m_dim[0]; i++){
                std::cout << m_data[i];
                if(i != m_dim[0]-1) std::cout << ", ";
            }
            std::cout << "]";
        }else{
            const std::vector<u64>& m_dim = t.m_dim;
            std::cout << "[";
            for(u64 i = 0; i < m_dim[0]; i++){
                if(i > 0) std::cout << std::string(toprank - t.rank() + 1, ' ');
                std::cout << t(i);
                if(i != m_dim[0]-1) std::cout << std::string(t.rank()-2, '\n');
            }
            std::cout << "]";
        }

        return out;
    }

    /*
     * TODO: Neex to FIX this. We need tensor template too
    template <typename dt>
    template <typename index_t, typename... indices_t>
    inline Tensor<> operator () (index_t index, indices_t... indices){

    }
    */

    template <typename dt>
    inline std::iostream& operator >> (std::iostream& in, Tensor<dt>& m){
        dt* m_data = m.m_data;
        u64 m_nelem = m.nr * m.nc;

        // elements are taken in row major order
        for(u64 i = 0; i < m_nelem; i++){
            in >> m_data[i];
        }

        return in;
    }
}

#endif // TENSORIMPL_H_
