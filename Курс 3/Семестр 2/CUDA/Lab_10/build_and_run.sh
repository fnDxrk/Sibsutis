#!/bin/bash

set -e

ARCH="-arch=sm_89"

# Компиляция matrix_transpose.cu
echo "▶️  [1/2] Компиляция matrix_transpose..."
nvcc $ARCH -o matrix_transpose/matrix_transpose matrix_transpose/matrix_transpose.cu

# Компиляция vector_product.cu
echo "▶️  [2/2] Компиляция vector_product..."
nvcc $ARCH -o vector_product/vector_product vector_product/vector_product.cu

echo "✅ Компиляция завершена."
echo ""

# Запуск matrix_transpose
echo "▶️  Запуск matrix_transpose:"
./matrix_transpose/matrix_transpose
echo ""

# Запуск vector_product
echo "▶️  Запуск vector_product:"
./vector_product/vector_product
echo ""
