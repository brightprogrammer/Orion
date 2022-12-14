#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "Typedefs.hpp"

namespace Orion{

    /**
     * Tensor is a general linear algebra object in Orion.
     * dt = int, float, double etc...
     * */
    template <typename dt>
    class Tensor {
    public:
        Tensor() = default;

        /**
         * Constructure for Tensor class.
         *
         * @param dim specifies the dimensions of tensor to be created.
         * eg : a tensor with dim = {2, 2, 2} will be a 2x2x2 cube.
         * */
        Tensor(const DimVec& dim);

        /**
         * Create tensor with given dimensions and use given data.
         * User is responsible for keeping this data valid. Tensor
         * class doesn't make a copy of this data anywhere.
         * */
        Tensor(const DimVec& dim, dt* data);

        /**
         * Fill all elements with zero. This will basically memset the whole data array.
         * Operation is applied in place.
         * */
        inline void zeroes();

        /**
         * Fill all elements with given value. This is an inplace operation.
         *
         * @param x is the value to be filled.
         * */
        inline void fill(dt x);

        /**
         * Fill all elements with random values. This is an inplace operation.
         *
         * @param max Maximum value of random.
         * @param min Minimum value of random.
         * */
        inline void randomize(dt min, dt max);

        /**
         * Get dimension vector of this Tensor.
         * @return DimVec
         * */
        inline const DimVec& dim() const { return m_dim; }

        /**
         * Get direct access to data of Tensor.
         * @return dt*
         * */
        inline dt* data() { return m_data; }

        /**
         * Get total number of scalar elements in tensor
         * @return u64 product of all elements of value returned by dim.
         * */
        inline u64 nelem() const { return m_nelem; }

        /**
         * Get rank of this tensor.
         * @return u64 rank of tensor.
         * */
        inline u64 rank() const { return static_cast<u64>(m_dim.size()); }

        /**
         * Print Tensor as a linear array, i.e a Tensor of rank 1.
         * */
        inline void printLinear() const;

        /**
         * To be used when the tensor is of rank 0, i.e a scalar.
         * */
        inline dt value() { if(rank() != 0) std::cerr << "WARN : getting value for non scalar tensor!\n"; return m_scalar; }

        /**
         * Tensor operator to get scalar or subtensor
         * */
        template<typename index_t = u64>
        inline Tensor<dt> operator () (index_t index) const{
            /*
             * If rank of tensor is 1 then a rank 0 tensor will be returned
             * i.e a scalar
             * */
            if(rank() == 1){
                Tensor<dt> res;
                res.m_scalar = m_data[index];
                return res;
            }else{
                dt* _data = m_data + m_nelem*index/m_dim[0];
                Tensor<dt> res(DimVec(m_dim.begin()+1, m_dim.end()), _data);
                return res;
            }
        }

        /**
         * Tensor operator to get a subtensor at given indices.
         * Tensors can be thought of as a tree and indexing a tensor is
         * basically indexing nodes of this tree.
         *
         * This is a variadic operator that recursively returns subtensors.
         * This makes indexing lot easier.
         * */
        template <typename index_t, typename... indices_t>
        inline Tensor<dt> operator () (index_t index, indices_t... indices) const{
            dt* _data = m_data + m_nelem*index/m_dim[0];
            Tensor<dt> res(DimVec(m_dim.begin()+1, m_dim.end()), _data);
            return res(indices...);
        }

        template <typename _dt>
        inline friend std::ostream& operator << (std::ostream& out, const Tensor<_dt>& m);

        template <typename _dt>
        inline friend std::iostream& operator >> (std::iostream& in, Tensor<_dt>& m);

        inline Tensor<dt> operator + (const Tensor<dt>& m);
        inline Tensor<dt> operator - (const Tensor<dt>& m);
        inline Tensor<dt> operator * (const Tensor<dt>& m);
    private:
        union {
            dt* m_data; // we store data of tensor as linear array
            dt m_scalar; // if this is tensor of rank 0 then only scalar field is used
        };

        // stores information about tensor dimensions
        DimVec m_dim;

        // total number of elements in matrix
        u64 m_nelem = 0;
    };

} // namespace orion

#include "TensorImpl.hpp"

#endif // TENSOR_H_
