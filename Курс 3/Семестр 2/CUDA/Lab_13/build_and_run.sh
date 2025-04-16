#!/bin/bash

set -e

ARCH="-arch=sm_89"

echo "▶️  [1/2] Компиляция cublas_example..."
nvcc $ARCH -lcublas -o cublas_example cublas_example.cu

echo "▶️  [2/2] Компиляция wmma_example..."
nvcc $ARCH -lcublas -o wmma_example wmma_example.cu

echo "✅ Компиляция завершена."
echo ""

echo "▶️  Запуск cublas_example:"
./cublas_example
echo ""

echo "▶️  Запуск wmma_example:"
./wmma_example
echo ""