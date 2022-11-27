#include "src/Tensor.hpp"
#include <vector>

using namespace std;

int main() {
    Orion::Tensor<float> t({3,3,3,3,4});
    t.randomize(-10, 10);

    cout << t << "\n";

    cout << "\n";

    return 0;
}
