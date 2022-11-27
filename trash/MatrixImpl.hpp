#ifndef MATRIX_IMPL_H_
#define MATRIX_IMPL_H_

#include "Matrix.hpp"
#include <iomanip>

namespace Orion {

    template <typename dt>
    Matrix<dt>::Matrix(u64 nr, u64 nc) : nr(nr), nc(nc){
        data = reinterpret_cast<dt*>(malloc(sizeof(dt) * nr * nc));
    }

    template <typename dt>
    void Matrix<dt>::printLinear(){
        u64 nelem = nr * nc;

        // elements are taken in row major order
        for(u64 i = 0; i < nelem; i++){
            std::cout << data[i] << " ";
        }
    }


    template <typename dt>
    void Matrix<dt>::zeroes(){
        memset(data, 0, sizeof(dt) * nr * nc);
    }

    template <typename dt>
    void Matrix<dt>::fill(dt a){
        u64 nelem = nr * nc;
        for(u64 i = 0; i < nelem; i++){
            data[i] = a;
        };
    }

    template <typename dt>
    void Matrix<dt>::random(dt max){
        u64 nelem = nr * nc;
        srand(time(nullptr));
        for(u64 i = 0; i < nelem; i++){
            if(max != 0) data[i] = (static_cast<dt>(rand())/static_cast<dt>(RAND_MAX)) * max;
            else data[i] = static_cast<dt>(rand());
        };
    }

    template <typename dt>
    std::ostream& operator << (std::ostream& out, const Matrix<dt>& m){
        dt* data = m.data;
        u64 nr = m.nr;
        u64 nc = m.nc;

        // treating matrix as row major
        for(u64 r = 0; r < nr; r++){
            for(u64 c = 0; c < nc; c++){
                out << std::setw(10) << data[r*nc + c] << " ";
            }
            out << "\n";
        }

        return out;
    }

    template <typename dt>
    std::iostream& operator >> (std::iostream& in, Matrix<dt>& m){
        dt* data = m.data;
        u64 nelem = m.nr * m.nc;

        // elements are taken in row major order
        for(u64 i = 0; i < nelem; i++){
            in >> data[i];
        }

        return in;
    }

    template <typename dt>
    Matrix<dt> Matrix<dt>::operator + (const Matrix<dt>& m){
        assert(m.nr == nr && m.nc == nc && "MATRIX DIMENSIONS MUST BE EXACTLY SAME!");

        Matrix<dt> res(nr, nc);
        dt* res_data = res.data;
        dt* m_data = m.data;
        u64 nelem = nr*nc;

        for(u64 i = 0; i < nelem; i++){
            res_data[i] = data[i] + m_data[i];
        }

        return res;
    }

    template <typename dt>
    Matrix<dt> Matrix<dt>::operator - (const Matrix<dt>& m){
        assert(m.nr == nr && m.nc == nc && "MATRIX DIMENSIONS MUST BE EXACTLY SAME!");

        Matrix<dt> res(nr, nc);
        dt* res_data = res.data;
        dt* m_data = m.data;
        u64 nelem = nr*nc;

        for(u64 i = 0; i < nelem; i++){
            res_data[i] = data[i] - m_data[i];
        }

        return res;
    }

    template <typename dt>
    Matrix<dt> Matrix<dt>::operator * (const Matrix<dt>& m){
        assert(nc == m.nr && "MATRICES ARE NOT COMPATIBLE FOR MULTIPLICATION");
    }

} // namespace orion

#endif // MATRIX_IMPL_H_
