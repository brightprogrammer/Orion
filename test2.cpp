#include "src/Tensor.hpp"

using TF = Orion::Tensor<float>;

using namespace std;
int main(){
    TF a({3, 4});
    a.randomize(-1, 1);
    
    cout << "A:\n" << a << endl;
    
    TF b({4, 5});
    b.randomize(-1, 1);

    cout << "B:\n" << b << endl;

    TF c = a*b;

    cout << "C:\n" << c << endl;
    return 0;
}