#ifndef OPERATOR_H
#define OPERATOR_H

#include "expressions.hpp"

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
    inline auto operator-(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::minus<>{});
    }

    template<typename E1, typename E2>
    inline auto operator*(TensorBase<E1> const& u, TensorBase<E2> const& v){
        return BinaryExpr(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v), std::multiplies<>{});
    }

    template<typename E1, typename Scalar, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator*(TensorBase<E1> const& u, Scalar v){
        return BinaryScalarExpr(*static_cast<const E1*>(&u), v, std::multiplies<>{});
    }
    template<typename E1, typename Scalar, std::enable_if_t<std::is_scalar<Scalar>::value, bool> = true>
    inline auto operator*(Scalar v, TensorBase<E1> const& u){
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

    struct expo{
        template<typename T>
        constexpr auto operator()(const T& x) const{
            return std::exp(x);
        }
    };

    template<typename E1>
    inline auto exp_t(TensorBase<E1> const& u){
        return UnaryExpr(*static_cast<const E1*>(&u), expo{});
    }

}

#endif