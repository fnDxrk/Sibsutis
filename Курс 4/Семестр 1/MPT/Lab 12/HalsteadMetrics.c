#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define S 18.0

typedef struct { int val, idx; } Min1D;
typedef struct { int val, row, col; } Min2D;

typedef struct {
    int eta1, eta2, eta, N1, N2, N;
    double N_hat, V_star, V, L, L_hat, I, T1, T2, T3;
} Metrics;

Min1D task1_find_min(int* arr, int n) {
    Min1D res = {arr[0], 0};
    int i;
    for (i = 1; i < n; i++) {
        if (arr[i] < res.val) {
            res.val = arr[i];
            res.idx = i;
        }
    }
    return res;
}

void task2_bubble_sort(int* arr, int n) {
    int i, j, tmp;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}

int task3_binary_search(int* arr, int n, int target) {
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

Min2D task4_find_min_2d(int** matrix, int rows, int* cols) {
    Min2D res = {matrix[0][0], 0, 0};
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols[i]; j++) {
            if (matrix[i][j] < res.val) {
                res.val = matrix[i][j];
                res.row = i;
                res.col = j;
            }
        }
    }
    return res;
}

void task5_reverse_array(int* arr, int n) {
    int i, tmp;
    for (i = 0; i < n / 2; i++) {
        tmp = arr[i];
        arr[i] = arr[n - 1 - i];
        arr[n - 1 - i] = tmp;
    }
}

void task6_cyclic_shift_left(int* arr, int n, int k) {
    k %= n;
    int* tmp = malloc(n * sizeof(int));
    int i;
    for (i = 0; i < n; i++) tmp[i] = arr[(i + k) % n];
    for (i = 0; i < n; i++) arr[i] = tmp[i];
    free(tmp);
}

void task7_replace_value(int* arr, int n, int old_val, int new_val) {
    int i;
    for (i = 0; i < n; i++) {
        if (arr[i] == old_val) arr[i] = new_val;
    }
}

Metrics compute_metrics(int eta1, int eta2, int N1, int N2, int eta2_star) {
    Metrics m;
    int eta = eta1 + eta2;
    int N = N1 + N2;
    
    m.eta1 = eta1;
    m.eta2 = eta2;
    m.eta = eta;
    m.N1 = N1;
    m.N2 = N2;
    m.N = N;
    
    m.N_hat = (eta1 > 0 ? eta1 * log2(eta1) : 0) + (eta2 > 0 ? eta2 * log2(eta2) : 0);
    m.V_star = (2 + eta2_star) * log2(2 + eta2_star);
    m.V = (eta > 0 && N > 0) ? N * log2(eta) : 0;
    m.L = (m.V > 0) ? m.V_star / m.V : 0;
    m.L_hat = (eta1 > 0 && N2 > 0) ? (2.0 / eta1) * (eta2 / (double)N2) : 0;
    m.I = (eta1 > 0 && N1 > 0 && eta > 0) ? (2.0 / eta1) * (N2 / (double)N1) * eta * log2(eta) : 0;
    m.T1 = m.V_star / S;
    m.T2 = (m.N_hat * (eta1 * log2(eta2) + eta2 * log2(eta1))) / (2 * S);
    m.T3 = (N1 * N2 * log2(eta)) / (2 * S);
    
    return m;
}

const char* task_names[] = {
    "1. Минимальный элемент одномерного массива и его индекс",
    "2. Сортировка пузырьком", 
    "3. Бинарный поиск",
    "4. Минимальный элемент двумерного массива",
    "5. Перестановка в обратном порядке",
    "6. Циклический сдвиг влево на k позиций",
    "7. Замена всех вхождений значения"
};

int eta2_stars[] = {3, 3, 3, 4, 2, 3, 4};

int main() {
    printf("====================================================================================================\n");
    printf("ЛАБОРАТОРНАЯ РАБОТА №12: МЕТРИКИ ХОЛСТЕДА (C)\n");
    printf("====================================================================================================\n");
    
    int task_params[7][3] = {
        {7, 6, 13}, {16, 32, 24}, {14, 24, 20}, {17, 36, 28},
        {11, 12, 10}, {17, 28, 22}, {12, 14, 12}
    };
    
    double sum_l_hat_v = 0, sum_v_sq = 0;
    
    for (int i = 0; i < 7; i++) {
        Metrics m = compute_metrics(task_params[i][0], task_params[i][1], 
                                   task_params[i][0], task_params[i][1], eta2_stars[i]);
        
        printf("\n--- ЗАДАЧА %d: %s ---\n", i+1, task_names[i]);
        printf("η₂* (смысловые параметры): %d\n", eta2_stars[i]);
        printf("η₁ (уникальные операторы): %d\n", m.eta1);
        printf("η₂ (уникальные операнды): %d\n", m.eta2);
        printf("η (словарь): %d\n", m.eta);
        printf("N₁ (вхождения операторов): %d\n", m.N1);
        printf("N₂ (вхождения операндов): %d\n", m.N2);
        printf("N (длина реализации): %d\n", m.N);
        printf("Ń (предсказанная длина): %.2f\n", m.N_hat);
        printf("V* (потенциальный объём): %.2f\n", m.V_star);
        printf("V (объём реализации): %.2f\n", m.V);
        printf("L (уровень через V*): %.4f\n", m.L);
        printf("L̂ (уровень по реализации): %.4f\n", m.L_hat);
        printf("I (интеллектуальное содержание): %.2f\n", m.I);
        printf("T̂₁ (время по V*): %.2f сек\n", m.T1);
        printf("T̂₂ (время по Ń): %.2f сек\n", m.T2);
        printf("T̂₃ (время по реализации): %.2f сек\n", m.T3);
        
        sum_l_hat_v += m.L_hat * m.V;
        sum_v_sq += m.V * m.V;
    }
    
    double lambda1 = sum_l_hat_v / 14.0;
    double lambda2 = sum_v_sq / 14.0;
    
    printf("\n====================================================================================================\n");
    printf("СРЕДНИЕ ЗНАЧЕНИЯ УРОВНЕЙ ЯЗЫКА ПРОГРАММИРОВАНИЯ:\n");
    printf("λ₁ = (Σ L̂ᵢ·Vᵢ) / (2·n) = %.2f (Python: 5.52)\n", lambda1);
    printf("λ₂ = (Σ Vᵢ²) / (2·n) = %.2f (Python: 29906.42)\n", lambda2);
    printf("====================================================================================================\n");
    
    return 0;
}

