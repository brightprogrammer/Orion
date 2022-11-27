#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>

typedef uint32_t u32;
typedef uint64_t u64;

namespace Orion {

    /**
     * Simple matrix implementation class.
     * This class treats created matrices as row major (for now).
     * */
    template <typename dt>
    class Matrix {
    public:
        Matrix() = default;
        Matrix(u64 nr, u64 nc);

        void zeroes();
        void random(dt rand_max = 0);
        void fill(dt a);

        // mostly for debugging purposes
        void printLinear();

        template <typename _dt>
        friend std::ostream& operator << (std::ostream& out, const Matrix<_dt>& m);

        template <typename _dt>
        friend std::iostream& operator >> (std::iostream& in, Matrix<_dt>& m);

        Matrix<dt> operator + (const Matrix<dt>& m);
        Matrix<dt> operator - (const Matrix<dt>& m);
        Matrix<dt> operator * (const Matrix<dt>& m);

    private:
        dt* data = nullptr;
        u64 nr = 0, nc = 0;
    }; // class Matrix

} // namespace orion

#include "MatrixImpl.hpp"

#endif // MATRIX_H_
