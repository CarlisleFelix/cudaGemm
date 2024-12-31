#include <iostream>
#include <random>

int main(){
    std::random_device rd; // 获取硬件随机数种子
    std::mt19937 gen(rd()); // 使用梅森旋转算法生成随机数
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f); // 生成0到1之间的随机浮点数
    for (int i = 0; i < 20 ; i++) {
        float tmp = dis(gen);
        std::cout << tmp << std::endl;
    }
    return 0;
}