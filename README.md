# Feynman-cpp
Feynman Machine implementation in C++ (port of OgmaNeo code)

To study the functioning of a Feynman machine, described [here](https://arxiv.org/abs/1609.03971) and found [here](https://github.com/ogmacorp/OgmaNeo), I needed functioning c++ code with no extras. I took the code from [Ogma Intelligent Systems Corp] (https://ogma.ai/) and removed everything that was not essential.

I'm interested in the following questions:
* Can this approach be properly vectorised (for AVX2 and AVX-512)
* Could this approach be run in parallel (with 8 threads and shared memory) on a regular CPU
* Could this approach be run in parallel (with 200 threads, NUMA) on Xeon Phi’s
