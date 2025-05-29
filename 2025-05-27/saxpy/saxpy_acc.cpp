#include <iostream>
#include <vector>

int main() {
    int n = 1 << 20;
    float a = 2.0f;
    std::vector<float> x(n, 1.0f), y(n, 2.0f);

    #pragma acc data copyin(x[0:n]) copy(y[0:n])
    {
        #pragma acc parallel loop
        for (int i = 0; i < n; ++i)
            y[i] = a * x[i] + y[i];
    }

    std::cout << "y[0] = " << y[0] << std::endl;
    return 0;
}
