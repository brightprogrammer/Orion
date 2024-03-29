#ifndef OPERATOR_H
#define OPERATOR_H

#include "Expressions.hpp"

#include <functional>
#include <cmath>

namespace Orion{

    template<typename E1, typename E2>
    inline auto operator+(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::plus<>{});
    }

    template<typename E1, typename Scalar, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator+(TensorBase<E1> const& u, Scalar v){
    return BinaryScalarExpr(*static_cast<const E1*>(&u), v, std::plus{});
    }

    template<typename E1, typename Scalar, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator+(Scalar v, TensorBase<E1> const& u){
        return u + v;
    }

    template<typename E1, typename E2>
    inline auto operator+=(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return ;
    }


    template<typename E1, typename E2>
    inline auto operator-(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::minus<>{});
    }

    //operator%- element-wise multiplication.

    template<typename E1, typename E2>
    inline auto operator%(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::multiplies<>{});
    }

    template<typename E1, typename Scalar, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator%(TensorBase<E1> const& u, Scalar v){
        return BinaryScalarExpr(*static_cast<const E1*>(&u), v, std::multiplies<>{});
    }
    template<typename E1, typename Scalar, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator%(Scalar v, TensorBase<E1> const& u){
        return u*v;
    }

    using namespace std::placeholders;
    template<typename E1, typename Scalar, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator-(TensorBase<E1> const& u, Scalar v){
        auto minus_scalar = std::bind(std::minus<>{}, _1, v);
        return UnaryExpr(*static_cast<const E1*>(&u), minus_scalar);
    }

    template<typename Scalar, typename E1, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator-(Scalar u, TensorBase<E1> const& v){
        auto minus_scalar = std::bind(std::minus<>{}, u, _1);
        return UnaryExpr(*static_cast<const E1*>(&v), minus_scalar);
    }

    template<typename E1, typename E2>
    inline auto operator==(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::equal_to<>{});
    }
    template<typename E1, typename E2>
    inline auto operator>=(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::greater_equal<>{});
    }
    template<typename E1, typename E2>
    inline auto operator<=(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::less_equal<>{});
    }
    template<typename E1, typename E2>
    inline auto operator<(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::less<>{});
    }
    template<typename E1, typename E2>
    inline auto operator>(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::greater<>{});
    }
    
    struct expo{
        template<typename T>
        inline auto operator()(const T& x) const{
            return std::exp(x);
        }
    };

    template<typename E1>
    inline auto exp_t(TensorBase<E1> const& u){
        return UnaryExpr(*static_cast<const E1*>(&u), expo{});
    }

    struct pow_t{
        int _p;
        pow_t(int p) : _p(p){}

        template<typename T>
        inline auto operator()(const T& x) const{
            return std::pow(x, _p);
        }
    };

    template<typename E1>
    inline auto pow(TensorBase<E1> const& u,const int p){
        return UnaryExpr(*static_cast<const E1*>(&u), pow_t(p));
    }

    template<typename E1, typename E2>
    inline auto operator*(TensorBase<E1> const& u, TensorBase<E2> const& v){
        assert(u.rank() == 2 && v.rank() == 2);
        auto& s1 = u.dim();
        auto& s2 = v.dim();
        assert(s1[1] == s2[0]);
        static_assert(std::is_same<typename E1::value_type,typename E2::value_type>::value, "Matmul: Different element types!");
        Tensor<typename E1::value_type> t({s1[0], s2[1]});
        auto data = t.data();
        for(u64 i = 0; i < s1[0]; i++){
            for(u64 j = 0; j < s2[1];j++){
                u64 pos = i*s2[1]+j;
                data[pos] = 0;
                for(u64 k = 0; k < s1[1]; k++){
                    data[pos] += u[i*s1[1] + k]*v[k*s2[1]+j];
                }
            }
        }
        return t;
    }

}



#endif