#!/bin/bash

set -e

ARCH="-arch=sm_89"

echo "▶️  [1/5] Компиляция CUDA Runtime API (cuda_api)..."
nvcc $ARCH -o cuda_api/matrix_multiplication cuda_api/matrix_multiplication.cu

echo "▶️  [2/5] Компиляция PTX для CUDA Driver API (cuda_driver_api)..."
nvcc $ARCH -ptx cuda_driver_api/kernel/kernel.cu -o cuda_driver_api/ptx/kernel.ptx

echo "▶️  [3/5] Компиляция оболочки Driver API (matrix_multiplication.cu)..."
nvcc $ARCH -lcuda -o cuda_driver_api/matrix_multiplication cuda_driver_api/matrix_multiplication.cu

echo "✅ Компиляция завершена."
echo ""

echo "▶️  Запуск CUDA Runtime API:"
./cuda_api/matrix_multiplication
echo ""

echo "▶️  Запуск CUDA Driver API:"
./cuda_driver_api/matrix_multiplication
echo ""

echo "▶️  Запуск Numba:"
python3 numba/matrix_multiplication.py
echo ""

echo "▶️  Запуск PyCUDA #1:"
python3 pycuda/matrix_multiplication1.py
echo ""

echo "▶️  Запуск PyCUDA #2:"
python3 pycuda/matrix_multiplication2.py
echo ""
