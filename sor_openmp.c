/*
 * =============================================================
 *  Метод на крайните елементи (МКЕ) – Паралелна версия
 *  Итерационен метод: Successive Over-Relaxation (SOR)
 *  Паралелизъм: OpenMP (Red-Black ordering)
 * =============================================================
 *
 * Задача: 2D уравнение на Поасон с нулеви гранични условия
 *   -∇²u = f(x,y)   в  Ω = [0,1] × [0,1]
 *       u = 0        на ∂Ω  (Дирихле)
 *
 * Паралелна стратегия – Red-Black (шахматна) наредба:
 *   При стандартна SOR, новата стойност u[i][j] зависи от
 *   стойности от текущата и предишната итерация → RACE CONDITION
 *   при наивна паралелизация.
 *
 *   Red-Black решение:
 *   - "Червени" точки: (i+j) % 2 == 0  (независими помежду си)
 *   - "Черни"  точки: (i+j) % 2 == 1  (независими помежду си)
 *   Всяка стъпка: обнови ВСИЧКИ червени → после ВСИЧКИ черни.
 *   Вътре в цвят → пълна независимост → безопасен OpenMP.
 *
 * Компилация:
 *   gcc -O2 -fopenmp -o sor_omp sor_openmp.c -lm
 *
 * Употреба:
 *   OMP_NUM_THREADS=8 ./sor_omp [N] [omega] [max_iter] [tol]
 *   Пример: OMP_NUM_THREADS=16 ./sor_omp 512 1.9 10000 1e-6
 * =============================================================
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define U(i,j)  u[(i)*(N+1) + (j)]
#define F(i,j)  f[(i)*(N+1) + (j)]

static double f_rhs(double x, double y)
{
    return 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}
static double u_exact(double x, double y)
{
    return sin(M_PI * x) * sin(M_PI * y);
}

int main(int argc, char *argv[])
{
    int    N        = (argc > 1) ? atoi(argv[1]) : 512;
    double omega    = (argc > 2) ? atof(argv[2]) : 1.9;
    int    max_iter = (argc > 3) ? atoi(argv[3]) : 10000;
    double tol      = (argc > 4) ? atof(argv[4]) : 1e-6;

    if (N < 3 || omega <= 0.0 || omega >= 2.0) {
        fprintf(stderr, "Грешни параметри: N >= 3, omega в (0,2)\n");
        return 1;
    }

    int    n  = N + 1;
    double h  = 1.0 / N;
    double h2 = h * h;

    printf("=== SOR  (Паралелна OpenMP версия) ===\n");
    printf("Мрежа: %d x %d   h = %g\n", n, n, h);
    printf("omega = %.4f   max_iter = %d   tol = %g\n", omega, max_iter, tol);
    printf("OpenMP нишки: %d\n\n", omp_get_max_threads());

    double *u = (double *)calloc((size_t)n * n, sizeof(double));
    double *f = (double *)malloc((size_t)n * n * sizeof(double));
    if (!u || !f) { fprintf(stderr, "Грешка при заделяне на памет\n"); return 1; }

    /* ---- инициализация ---- */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i <= N; i++)
        for (int j = 0; j <= N; j++) {
            F(i,j) = f_rhs(i * h, j * h);
            U(i,j) = 0.0;
        }

    double t_start = omp_get_wtime();

    int    iter = 0;
    double res  = tol + 1.0;

    while (iter < max_iter && res > tol) {

        double res_red   = 0.0;
        double res_black = 0.0;

        /* ====================================================
         *  ЧЕРВЕНА стъпка: обновяване на (i+j) четни точки
         *  Всички червени точки са независими – безопасен OMP
         * ==================================================== */
        #pragma omp parallel for collapse(2) schedule(static) \
                                 reduction(+:res_red)
        for (int i = 1; i < N; i++) {
            for (int j = 1; j < N; j++) {
                if ((i + j) % 2 != 0) continue;
                double u_gs = 0.25 * (U(i-1,j) + U(i+1,j)
                                    + U(i,j-1) + U(i,j+1)
                                    + h2 * F(i,j));
                double delta = omega * (u_gs - U(i,j));
                U(i,j) += delta;
                res_red += delta * delta;
            }
        }

        /* ====================================================
         *  ЧЕРНА стъпка: обновяване на (i+j) нечетни точки
         *  Вече виждат актуалните червени стойности
         * ==================================================== */
        #pragma omp parallel for collapse(2) schedule(static) \
                                 reduction(+:res_black)
        for (int i = 1; i < N; i++) {
            for (int j = 1; j < N; j++) {
                if ((i + j) % 2 != 1) continue;
                double u_gs = 0.25 * (U(i-1,j) + U(i+1,j)
                                    + U(i,j-1) + U(i,j+1)
                                    + h2 * F(i,j));
                double delta = omega * (u_gs - U(i,j));
                U(i,j) += delta;
                res_black += delta * delta;
            }
        }

        res = sqrt(res_red + res_black);
        iter++;

        if (iter % 500 == 0)
            printf("  iter %5d   residual = %e\n", iter, res);
    }

    double t_end = omp_get_wtime();

    printf("\n--- Резултати ---\n");
    printf("Итерации       : %d\n", iter);
    printf("Краен residual : %e\n", res);
    printf("Време (сек.)   : %.4f\n", t_end - t_start);

    /* ---- L2 грешка ---- */
    double err = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:err)
    for (int i = 1; i < N; i++)
        for (int j = 1; j < N; j++) {
            double d = U(i,j) - u_exact(i * h, j * h);
            err += d * d;
        }
    printf("L2 грешка      : %e\n", h * sqrt(err));

    free(u);
    free(f);
    return 0;
}
