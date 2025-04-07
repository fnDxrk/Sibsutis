#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK_CUDA_ERROR(call)                                                                      \
    {                                                                                               \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            std::cerr << "CUDA error in " << #call << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    }

// Параметры сетки и сферы
const int Nx = 64, Ny = 64, Nz = 64;
const float x_min = -1.0f, x_max = 1.0f;
const float y_min = -1.0f, y_max = 1.0f;
const float z_min = -1.0f, z_max = 1.0f;
const int Ntheta = 128, Nphi = 256;
const float radius = 1.0f;

// Константы для передачи в ядра
__constant__ float c_x_min, c_x_max, c_y_min, c_y_max, c_z_min, c_z_max;
__constant__ int c_Nx, c_Ny, c_Nz;
__constant__ float c_dx, c_dy, c_dz;

// Тестовая функция
float test_function(float x, float y, float z)
{
    return x * y * z; // Интеграл должен быть 0 для симметрии
}

// Инициализация данных сетки
void init_grid(std::vector<float>& grid)
{
    float dx = (x_max - x_min) / (Nx - 1);
    float dy = (y_max - y_min) / (Ny - 1);
    float dz = (z_max - z_min) / (Nz - 1);
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                float x = x_min + i * dx;
                float y = y_min + j * dy;
                float z = z_min + k * dz;
                grid[i + j * Nx + k * Nx * Ny] = test_function(x, y, z);
            }
        }
    }
}

// Ядро для метода с текстурной памятью
__global__ void kernel_texture(cudaTextureObject_t tex_obj, float* d_result)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = 0.0f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < Ntheta && idy < Nphi) {
        float theta = idx * (M_PI / Ntheta);
        float phi = idy * (2.0f * M_PI / Nphi);
        float sin_theta = sinf(theta);

        float x = radius * sin_theta * cosf(phi);
        float y = radius * sin_theta * sinf(phi);
        float z = radius * cosf(theta);

        // Нормализация для текстуры (0..1)
        float tex_x = (x - c_x_min) / (c_x_max - c_x_min) * (c_Nx - 1);
        float tex_y = (y - c_y_min) / (c_y_max - c_y_min) * (c_Ny - 1);
        float tex_z = (z - c_z_min) / (c_z_max - c_z_min) * (c_Nz - 1);

        float f = tex3D<float>(tex_obj, tex_x + 0.5f, tex_y + 0.5f, tex_z + 0.5f); // Смещение на 0.5 для центрирования
        sdata[tid] += f * radius * radius * sin_theta * (M_PI / Ntheta) * (2.0f * M_PI / Nphi);
    }
    __syncthreads();

    // Редукция в блоке
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_result, sdata[0]);
}

// Ядро для метода без текстур (ступенчатая интерполяция)
__global__ void kernel_stepwise(float* d_grid, float* d_result)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = 0.0f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < Ntheta && idy < Nphi) {
        float theta = idx * (M_PI / Ntheta);
        float phi = idy * (2.0f * M_PI / Nphi);
        float sin_theta = sinf(theta);

        float x = radius * sin_theta * cosf(phi);
        float y = radius * sin_theta * sinf(phi);
        float z = radius * cosf(theta);

        int i = roundf((x - c_x_min) / c_dx);
        int j = roundf((y - c_y_min) / c_dy);
        int k = roundf((z - c_z_min) / c_dz);

        i = max(0, min(c_Nx - 1, i));
        j = max(0, min(c_Ny - 1, j));
        k = max(0, min(c_Nz - 1, k));

        float f = d_grid[i + j * c_Nx + k * c_Nx * c_Ny];
        sdata[tid] += f * radius * radius * sin_theta * (M_PI / Ntheta) * (2.0f * M_PI / Nphi);
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_result, sdata[0]);
}

// Ядро для метода без текстур (линейная интерполяция)
__global__ void kernel_linear(float* d_grid, float* d_result)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = 0.0f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < Ntheta && idy < Nphi) {
        float theta = idx * (M_PI / Ntheta);
        float phi = idy * (2.0f * M_PI / Nphi);
        float sin_theta = sinf(theta);

        float x = radius * sin_theta * cosf(phi);
        float y = radius * sin_theta * sinf(phi);
        float z = radius * cosf(theta);

        float fx = (x - c_x_min) / c_dx;
        float fy = (y - c_y_min) / c_dy;
        float fz = (z - c_z_min) / c_dz;

        int i = floorf(fx);
        float wx = fx - i;
        int j = floorf(fy);
        float wy = fy - j;
        int k = floorf(fz);
        float wz = fz - k;

        i = max(0, min(c_Nx - 2, i));
        j = max(0, min(c_Ny - 2, j));
        k = max(0, min(c_Nz - 2, k));

        float f000 = d_grid[i + j * c_Nx + k * c_Nx * c_Ny];
        float f001 = d_grid[i + j * c_Nx + (k + 1) * c_Nx * c_Ny];
        float f010 = d_grid[i + (j + 1) * c_Nx + k * c_Nx * c_Ny];
        float f011 = d_grid[i + (j + 1) * c_Nx + (k + 1) * c_Nx * c_Ny];
        float f100 = d_grid[(i + 1) + j * c_Nx + k * c_Nx * c_Ny];
        float f101 = d_grid[(i + 1) + j * c_Nx + (k + 1) * c_Nx * c_Ny];
        float f110 = d_grid[(i + 1) + (j + 1) * c_Nx + k * c_Nx * c_Ny];
        float f111 = d_grid[(i + 1) + (j + 1) * c_Nx + (k + 1) * c_Nx * c_Ny];

        float f0 = (1 - wz) * f000 + wz * f001;
        float f1 = (1 - wz) * f010 + wz * f011;
        float f2 = (1 - wz) * f100 + wz * f101;
        float f3 = (1 - wz) * f110 + wz * f111;

        float f01 = (1 - wy) * f0 + wy * f1;
        float f23 = (1 - wy) * f2 + wy * f3;

        float f = (1 - wx) * f01 + wx * f23;

        sdata[tid] += f * radius * radius * sin_theta * (M_PI / Ntheta) * (2.0f * M_PI / Nphi);
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_result, sdata[0]);
}

int main()
{
    // Инициализация данных
    std::vector<float> h_grid(Nx * Ny * Nz);
    init_grid(h_grid);

    // Установка констант
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_x_min, &x_min, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_x_max, &x_max, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_y_min, &y_min, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_y_max, &y_max, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_z_min, &z_min, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_z_max, &z_max, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_Nx, &Nx, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_Ny, &Ny, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_Nz, &Nz, sizeof(int)));
    float dx = (x_max - x_min) / (Nx - 1);
    float dy = (y_max - y_min) / (Ny - 1);
    float dz = (z_max - z_min) / (Nz - 1);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_dx, &dx, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_dy, &dy, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_dz, &dz, sizeof(float)));

    // Подготовка текстурной памяти
    cudaArray* d_array;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CHECK_CUDA_ERROR(cudaMalloc3DArray(&d_array, &channelDesc, make_cudaExtent(Nx, Ny, Nz)));
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(h_grid.data(), Nx * sizeof(float), Nx, Ny);
    copyParams.dstArray = d_array;
    copyParams.extent = make_cudaExtent(Nx, Ny, Nz);
    copyParams.kind = cudaMemcpyHostToDevice;
    CHECK_CUDA_ERROR(cudaMemcpy3D(&copyParams));

    // Создание объекта текстуры
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex_obj = 0;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr));

    // Подготовка глобальной памяти для второго метода
    float* d_grid;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grid, Nx * Ny * Nz * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_grid, h_grid.data(), Nx * Ny * Nz * sizeof(float), cudaMemcpyHostToDevice));

    // Результаты
    float *d_result_texture, *d_result_stepwise, *d_result_linear;
    float h_result_texture = 0.0f, h_result_stepwise = 0.0f, h_result_linear = 0.0f;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result_texture, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result_stepwise, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result_linear, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_result_texture, 0, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_result_stepwise, 0, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_result_linear, 0, sizeof(float)));

    // Конфигурация запуска
    dim3 block(16, 16);
    dim3 grid((Ntheta + block.x - 1) / block.x, (Nphi + block.y - 1) / block.y);

    // Измерение времени
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float time_texture, time_stepwise, time_linear;

    // Текстурная память
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernel_texture<<<grid, block>>>(tex_obj, d_result_texture);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_texture, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result_texture, d_result_texture, sizeof(float), cudaMemcpyDeviceToHost));

    // Ступенчатая интерполяция
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernel_stepwise<<<grid, block>>>(d_grid, d_result_stepwise);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_stepwise, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result_stepwise, d_result_stepwise, sizeof(float), cudaMemcpyDeviceToHost));

    // Линейная интерполяция
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernel_linear<<<grid, block>>>(d_grid, d_result_linear);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_linear, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result_linear, d_result_linear, sizeof(float), cudaMemcpyDeviceToHost));

    // Вывод результатов
    std::cout << std::left << std::setw(32) << "Texture Memory Result:"
              << std::setw(10) << std::fixed << std::setprecision(6) << h_result_texture
              << std::setw(5) << "Time: " << std::setprecision(6) << time_texture << " ms" << std::endl;

    std::cout << std::left << std::setw(32) << "Stepwise Interpolation Result:"
              << std::setw(10) << std::fixed << std::setprecision(6) << h_result_stepwise
              << std::setw(5) << "Time: " << std::setprecision(6) << time_stepwise << " ms" << std::endl;

    std::cout << std::left << std::setw(32) << "Linear Interpolation Result:"
              << std::setw(10) << std::fixed << std::setprecision(6) << h_result_linear
              << std::setw(5) << "Time: " << std::setprecision(6) << time_linear << " ms" << std::endl;

    // Очистка
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(tex_obj));
    CHECK_CUDA_ERROR(cudaFreeArray(d_array));
    CHECK_CUDA_ERROR(cudaFree(d_grid));
    CHECK_CUDA_ERROR(cudaFree(d_result_texture));
    CHECK_CUDA_ERROR(cudaFree(d_result_stepwise));
    CHECK_CUDA_ERROR(cudaFree(d_result_linear));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}
