#ifndef EXPRESSIONS_H
#define EXPRESSIONS_H

#include "Tensor.hpp"

namespace Orion
{
    template<typename E1, typename E2, typename Callable>
    class BinaryExpr : public TensorBase<BinaryExpr<E1, E2, Callable>>{
        static_assert(std::is_same<typename E1::value_type,typename E2::value_type>::value, "Cannot evaluate expression of different tensor elements.");
        
        
        E1 const& _u;
        E2 const& _v;
        Callable const&  callable;
        public:
            typedef typename E1::value_type value_type;

            // template<typename Func> 
            BinaryExpr(E1 const& u, E2 const& v, Callable const& func) : _u(u), _v(v), callable(func) {
                assert(u.rank() == v.rank());
                auto s1 = u.dim();
                auto s2 = v.dim();
                for(size_t i = 0; i < s1.size(); i++)
                    assert(s1[i] == s2[i]);
            }
            inline auto operator[](size_t i) const {
                return callable(_u[i], _v[i]);
            }
            size_t rank() const{
                return _v.rank();
            }
            const DimVec& dim() const{ return _u.dim(); } 
    };

    template<typename E1, typename Scalar, typename Callable, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    class BinaryScalarExpr : public TensorBase<BinaryScalarExpr<E1, Scalar, Callable>>{
        static_assert(std::is_same<typename E1::value_type, Scalar>::value, "Cannot evaluate expression of different tensor elements.");

        E1 const& _u;
        Scalar _v;
        Callable const& callable;

        public:
            typedef typename E1::value_type value_type;

            BinaryScalarExpr(E1 const& u, Scalar v, Callable const& func) : _u(u), _v(v), callable(func){
            }

            inline auto operator[](size_t i) const{
                return callable(_u[i], _v);
            }
            size_t rank() const{
                return _u.rank();
            }
            const DimVec& dim() const{ return _u.dim(); }
    };

    template<typename E1, typename Callable>
    class UnaryExpr : public TensorBase<UnaryExpr<E1, Callable>>{
        E1 const& _u;
        Callable callable;

        public:
            typedef typename E1::value_type value_type;

            UnaryExpr(E1 const& u, Callable const& func) : _u(u), callable(func) 
            {}

            inline auto operator[](size_t i) const{
                return callable(_u[i]);
            }
            size_t rank() const{
                return _u.rank();
            }
            const DimVec & dim() const{ return _u.dim(); } 
    };
} // namespace Orion
#endif