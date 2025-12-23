// On linux with g++, compile with
// g++ -O3 -mpclmul -msse4.1 crc.cpp crctest.cpp -o crctest.exe
// It took 9 seconds to run and pass on my computer (AMD Ryzen 7 5825U)
#include "crc.h"
#include "stdio.h"

int main(int argc, const char** argv)
{
    printf("crctest: %s\n", g_crc->SelfTest() ? "passed" : "failed");
    return 0;
}
