#!/bin/bash

echo "▶️  [1/3] Компиляция kernel.cu в kernel.ptx..."
nvcc -ptx kernel.cu -arch=sm_89 -o kernel.ptx

echo "▶️  [2/3] Компиляция cuda_api..."
nvcc -o cuda_api cuda_api.cu -arch=sm_89 -O3

echo "▶️  [3/3] Компиляция cuda_api_driver..."
nvcc -o cuda_api_driver cuda_api_driver.cu -arch=sm_89 -lcuda -O3

echo "✅ Компиляция завершена."

echo -e "\n▶️  Запуск cuda_api:"
./cuda_api

echo -e "\n▶️  Запуск cuda_api_driver:"
./cuda_api_driver