#ifndef GPUMODE_KERNELS_H
#define GPUMODE_KERNELS_H

#include <iostream>
#include <cuComplex.h>
#include <cufft.h>

#define NOT_SUPPORTED(x) { std::cerr << "Whoops, we don't support this on the GPU: " << x << std::endl; exit(1); }

#define checkCuda(err) __checkCuda(err, (char *)__FILE__, __LINE__)
inline cudaError_t __checkCuda(cudaError_t err, char *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "Error in calling CUDA operation in " << file << " at line " << line << std::endl;
    std::cerr << "Error was " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  return err;
}

#define checkCufft(err) __checkCufft(err, (char *)__FILE__, __LINE__)
inline cufftResult_t __checkCufft(const cufftResult_t err, const char *const file, const int line) {
  if (err != CUFFT_SUCCESS) {
    std::cerr << "Error calling a cuFFT operation in " << file << " at line " << line << std::endl;
    // TODO: should we convert err to a string? (it is an enum documented in
    // the cuFFT documentation - there doesn't seem to be an official errorcode
    // -> string conversion routine, but there is _cudaGetErrorEnum...)
    std::cerr << "Error was " << (int)err << std::endl;
    exit(1);
  }
  return err;
}

template <class T>
class GpuMemHelper {
public:
    GpuMemHelper(size_t nElems, cudaStream_t stream) : managed(true), nBytes(sizeof(T) * nElems), cuStream(stream) {
        cpuData = new T[nElems];
        checkCuda(cudaHostRegister(cpuData, nBytes, cudaHostRegisterPortable));
        checkCuda(cudaMallocAsync(&gpuData, nBytes, cuStream));
    }

    GpuMemHelper(T* hostPtr, size_t nElems, cudaStream_t stream) : managed(false), cpuData(hostPtr), nBytes(sizeof(T) * nElems), cuStream(stream) {
        checkCuda(cudaHostRegister(cpuData, nBytes, cudaHostRegisterPortable));
        checkCuda(cudaMallocAsync(&gpuData, nBytes, cuStream));
    }

    GpuMemHelper(size_t nElems, cudaStream_t stream, bool gpuOnly) : managed(false), cpuData(nullptr), nBytes(sizeof(T) * nElems), cuStream(stream) {
        checkCuda(cudaMallocAsync(&gpuData, nBytes, cuStream));

        if (!gpuOnly) {
            cpuData = new T[nElems];
            checkCuda(cudaHostRegister(cpuData, nBytes, cudaHostRegisterPortable));
        }
    }

    ~GpuMemHelper() {
        if (cpuData) {
            checkCuda(cudaHostUnregister(cpuData));

            if (managed) {
                delete[] cpuData;
                cpuData = nullptr;
            }
        }

        if (gpuData) {
            checkCuda(cudaFreeAsync(gpuData, cuStream));
            gpuData = nullptr;
        }
    }

    inline GpuMemHelper* copyToDevice() {
        checkCpuData();

        checkCuda(cudaMemcpyAsync(gpuData, cpuData, nBytes, cudaMemcpyHostToDevice, cuStream));

        return this;
    }

    inline GpuMemHelper* copyToHost() {
        checkCpuData();

        checkCuda(cudaMemcpyAsync(cpuData, gpuData, nBytes, cudaMemcpyDeviceToHost, cuStream));

        return this;
    }

    inline GpuMemHelper* sync() {
        checkCuda(cudaStreamSynchronize(cuStream));

        return this;
    }

    inline T* ptr() { return cpuData; }
    inline T* gpuPtr() { return gpuData; }

    inline size_t size() { return nBytes; }

private:
    T* cpuData;
    T* gpuData;
    cudaStream_t cuStream;
    bool managed;
    size_t nBytes;

    void checkCpuData() {
        if (!cpuData) {
            cout << "Attempt to use null cpuData in GpuMemHelper" << endl;
            exit(1);
        }
    }
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
