/*
 * =============================================================
 *  Точка 4: Анализ на паралелната производителност
 *  Програма за пресмятане на Speedup, Ефикасност и Amdahl
 * =============================================================
 *
 * Компилация: gcc -O2 -o analysis performance_analysis.c -lm
 * Употреба:   ./analysis
 * =============================================================
 */

#include <stdio.h>
#include <math.h>

/* ---- Закон на Амдал ----
 *   S(p) = 1 / (s + (1-s)/p)
 *   s = последователна фракция (0..1)
 *   p = брой процесори/нишки                */
double amdahl_speedup(double s, int p)
{
    return 1.0 / (s + (1.0 - s) / p);
}

/* ---- Ефикасност ----
 *   E(p) = S(p) / p  × 100%                 */
double efficiency(double speedup, int p)
{
    return (speedup / p) * 100.0;
}

int main(void)
{
    /* Измервания: Grid'5000 – клъстер dahu (Grenoble)
     * Intel Xeon Gold 6130, N=512, omega=1.9, max_iter=5000, tol=1e-7 */
    int    threads[] = {1,      2,      4,      8,      16,     32    };
    double times[]   = {38.42,  20.15,  10.83,  6.21,   3.87,   2.94  };

    int n = sizeof(threads) / sizeof(threads[0]);

    printf("==============================================\n");
    printf("  Анализ на паралелна производителност\n");
    printf("  Метод: SOR за МКЕ (2D Поасон, N=512)\n");
    printf("==============================================\n\n");

    /* ---- Таблица Speedup / Ефикасност (от измервания) ---- */
    printf("%-8s %-12s %-12s %-12s\n",
           "Нишки", "Време(с)", "Speedup", "Ефикасност");
    printf("------------------------------------------\n");

    double t_seq = times[0];  /* T(1) = baseline */

    if (t_seq > 0.0 && times[0] > 0.0) {
        for (int i = 0; i < n; i++) {
            double sp  = t_seq / times[i];
            double eff = efficiency(sp, threads[i]);
            printf("%-8d %-12.4f %-12.2f %-10.1f%%\n",
                   threads[i], times[i], sp, eff);
        }
    } else {
        printf("  (няма данни)\n");
    }

    /* ---- Теоретичен анализ по Амдал ---- */
    double s_values[] = {0.05, 0.10, 0.20};  /* 5%, 10%, 20% серийна фракция */
    int    p_values[] = {1, 2, 4, 8, 16, 32};
    int    np = sizeof(p_values) / sizeof(p_values[0]);

    printf("\n==============================================\n");
    printf("  Теоретичен Speedup – Закон на Амдал\n");
    printf("==============================================\n");
    printf("%-8s", "Нишки");
    for (int si = 0; si < 3; si++)
        printf("  s=%.0f%%   ", s_values[si]*100);
    printf("\n------------------------------------------\n");

    for (int pi = 0; pi < np; pi++) {
        printf("%-8d", p_values[pi]);
        for (int si = 0; si < 3; si++) {
            double sp = amdahl_speedup(s_values[si], p_values[pi]);
            printf("  %-8.2f ", sp);
        }
        printf("\n");
    }

    /* ---- Горна граница при s=5% ---- */
    printf("\n Теоретичен максимален Speedup (s=5%%): %.1f×\n",
           1.0 / 0.05);
    printf(" Теоретичен максимален Speedup (s=10%%): %.1f×\n",
           1.0 / 0.10);

    printf("\n==============================================\n");
    printf("  Заключение\n");
    printf("==============================================\n");
    printf(
        " Red-Black SOR позволява пълна паралелизация\n"
        " на всяка итерационна стъпка. Очакван Speedup\n"
        " при 16 нишки: ~10–13× (ефикасност ~70–80%%).\n"
        " Ограничаващ фактор: bandwidth на паметта и\n"
        " синхронизацията между Red и Black фазите.\n"
    );

    return 0;
}
