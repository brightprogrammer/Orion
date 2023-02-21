#include "src/Tensor.hpp"
#include <vector>

using namespace std;

using Tf = Orion::Tensor<float>;

int main() {
    Tf t({200,300,400});
    t.randomize(0, 1);
    
    // cout << t  << endl;

    Tf t1 = t(0, 0);
    for(int i = 1; i < 300; i++)
        t1 = t1 + t(0, i)*t(0, i-1);

    // cout << t1 << endl;

    Tf t2 = t(0);
    for(int i = 1; i < 200; i++)
        t2 = t2 + t(i)*t(i-1);

    // cout << t2 << endl;
    return 0;
}
