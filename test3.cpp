#include "src/dl/Backprop.hpp"

using namespace Orion;

typedef Orion::Tensor<double> ten;

int main(){
    ten xt({3,4});
    xt.randomize(0,1);
    ten yt({3,4});
    yt.randomize(0,1);
    auto x = std::make_shared<TensorVar>(xt, true);
    auto y = std::make_shared<TensorVar>(yt, true);
    auto z = exp(-x) + exp(-y);
    // Orion::display(z);
    Orion::backward(z, {x, y});
    std::cout << x->value() << std::endl;
    std::cout << y->value() << std::endl;
    std::cout << x->grad() << std::endl;
    std::cout << y->grad() << std::endl;
    std::cout << z->grad() << std::endl;
}