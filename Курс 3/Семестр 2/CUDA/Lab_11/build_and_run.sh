#!/bin/bash

set -e

ARCH="-arch=sm_89"

echo "▶️  [1/3] Компиляция mem_copy_test..."
nvcc $ARCH -o memory_copy/mem_copy_test memory_copy/mem_copy_test.cu

echo "▶️  [2/3] Компиляция vector_addition..."
nvcc $ARCH -o vector_addition/vector_addition vector_addition/vector_addition.cu

echo "▶️  [3/3] Компиляция vector_product..."
nvcc $ARCH -o vector_product/vector_product vector_product/vector_product.cu

echo "✅ Компиляция завершена."
echo ""

echo "▶️  Запуск mem_copy_test:"
./memory_copy/mem_copy_test
echo ""

echo "▶️  Запуск vector_addition:"
./vector_addition/vector_addition
echo ""

echo "▶️  Запуск vector_product:"
./vector_product/vector_product
echo ""
