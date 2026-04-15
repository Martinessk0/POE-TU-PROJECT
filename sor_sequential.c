/*
 * =============================================================
 *  Метод на крайните елементи (МКЕ) – Последователна версия
 *  Итерационен метод: Successive Over-Relaxation (SOR)
 * =============================================================
 *
 * Задача: 2D уравнение на Поасон с нулеви гранични условия
 *   -∇²u = f(x,y)   в  Ω = [0,1] × [0,1]
 *       u = 0        на ∂Ω  (Дирихле)
 *
 * Точно решение за тест:
 *   f(x,y) = 2π²·sin(πx)·sin(πy)  →  u(x,y) = sin(πx)·sin(πy)
 *
 * Дискретизация (5-точков шаблон, стъпка h = 1/N):
 *   -(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4·u[i][j]) / h² = f[i][j]
 *
 * SOR итерация (стандартна):
 *   u_new = (1-ω)·u_old + (ω/4)·(u[i-1][j] + u[i+1][j]
 *                                + u[i][j-1] + u[i][j+1] + h²·f[i][j])
 *
 * Оптималното ω за правоъгълна мрежа:
 *   ω_opt = 2 / (1 + sin(π·h))  ≈ 1.8–1.95  за малки h
 *
 * Компилация:
 *   gcc -O2 -o sor_seq sor_sequential.c -lm
 *
 * Употреба:
 *   ./sor_seq [N] [omega] [max_iter] [tol]
 *   Пример: ./sor_seq 256 1.9 10000 1e-6
 * =============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ---- макроси за индексиране на плосък масив ---- */
#define U(i,j)   u[(i)*(N+1) + (j)]
#define F(i,j)   f[(i)*(N+1) + (j)]

/* ---- дясна страна и точно решение ---- */
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
    int    N        = (argc > 1) ? atoi(argv[1]) : 128;
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

    printf("=== SOR  (Последователна версия) ===\n");
    printf("Мрежа: %d x %d   h = %g\n", n, n, h);
    printf("omega = %.4f   max_iter = %d   tol = %g\n\n",
           omega, max_iter, tol);

    /* ---- заделяне на памет ---- */
    double *u = (double *)calloc((size_t)n * n, sizeof(double));
    double *f = (double *)malloc((size_t)n * n * sizeof(double));
    if (!u || !f) { fprintf(stderr, "Грешка при заделяне на памет\n"); return 1; }

    /* ---- инициализация: RHS и гранични условия u=0 ---- */
    for (int i = 0; i <= N; i++)
        for (int j = 0; j <= N; j++) {
            F(i,j) = f_rhs(i * h, j * h);
            U(i,j) = 0.0;
        }

    /* ---- SOR итерации ---- */
    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    int    iter = 0;
    double res  = tol + 1.0;

    while (iter < max_iter && res > tol) {
        res = 0.0;
        for (int i = 1; i < N; i++) {
            for (int j = 1; j < N; j++) {
                double u_gs = 0.25 * (U(i-1,j) + U(i+1,j)
                                    + U(i,j-1) + U(i,j+1)
                                    + h2 * F(i,j));
                double delta = omega * (u_gs - U(i,j));
                U(i,j) += delta;
                res += delta * delta;
            }
        }
        res = sqrt(res);
        iter++;

        if (iter % 500 == 0)
            printf("  iter %5d   residual = %e\n", iter, res);
    }

    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec)
                   + (te.tv_nsec - ts.tv_nsec) * 1e-9;

    printf("\n--- Резултати ---\n");
    printf("Итерации       : %d\n", iter);
    printf("Краен residual : %e\n", res);
    printf("Време (сек.)   : %.4f\n", elapsed);

    /* ---- L2 грешка спрямо точното решение ---- */
    double err = 0.0;
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
