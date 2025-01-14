import sympy as sp  # Для работы с символьными вычислениями
import numpy as np  # Для генерации списка X для функций
import matplotlib.pyplot as plt  # Для отображения графиков
from math import factorial, e, pi  # Факториал, числа e и pi
from sympy.abc import t, theta  # Для использования в символьных вычислениях (буквы t и theta)
from sympy import oo  # Для вычисления несобственных интегралов


# Функции для полиномов
def legandr_poly(n: int):
    """Данная функция вычисляет полином Лежандра (I рода)
     Входные параметры: n - степень полинома
     Выходные параметры: полином Лежандра (в символьном виде) для
    заданного n"""
    # Данный полином можно представить в виде произведения 1/2^n*n! и производной порядка n от(t ^ 2 - 1) ^ n
    pn_first_mult = 1 / (2 ** n * factorial(n))
    pn_second_mult = sp.diff(f"(t^2-1)^{n}", t, n)
    return pn_first_mult * pn_second_mult


def chebyshev_poly(n: int):
    """Данная функция вычисляет полином Чебышева (I рода)
     Входные параметры: n - степень полинома
     Выходные параметры: полином Чебышева (в символьном виде) для
    заданного n"""

    # Данный полином можно представить в виде произведения (-2) ^ n * n! / (2 * n)!, (1 - t ^ 2) ^ 1 / 2,
    # и производной порядка n от (sqrt(1 - t ^ 2)) ^ 2 * n - 1
    # Вычислим все множители
    tn_first_mult = (-2) ** n * factorial(n) / factorial(2 * n)
    tn_second_mult = sp.sympify("sqrt(1-t**2)")
    tn_third_mult = sp.diff(f"sqrt(1-t**2)**(2*{n}-1)", t, n)  # Производная порядка n по t
    return sp.simplify(tn_first_mult * tn_second_mult * tn_third_mult)  # Упростить (simplify) возвращаемый результат


def lagger_poly(n: int):
    """Данная функция вычисляет полином Лагерра
     Входные параметры: n - степень полинома
     Выходные параметры: полином Лагерра (в символьном виде) для
    заданного n"""

    # Данный полином можно представить в виде произведения e^t/n!, и производной порядка n от t^n * e^-t
    # Вычислим все множители
    ln_first_mult = sp.sympify(f"{e}**(t)/{factorial(n)}")
    ln_second_mult = sp.diff(sp.sympify(f"t**({n})*{e}**(-t)"), t, n)
    return ln_first_mult * ln_second_mult


def legandr(xi, func: sp.core.mul.Mul):
    """ Данная функция раскладывает сигнал по 4 ненулевым членам ряда
    используя полиномы Лежандра.
     Функция отображает каждый элемент ряда на графике.
     Входные параметры: xi - список X, func - функция (в символьном
    виде)
     Выходные параметры: список Y"""

    n = 4  # Количество необходимых ненулевых членов ряда
    ft = 0  # Ряд
    i = 0  # Итератор
    eps = pow(10, -7)  # Используется для сравнения с нулём
    while i < n:
        legandr_polynome = legandr_poly(i)  # Нахождение полинома Лежандра
        cn = ((2 * i + 1) / 2) * sp.N(sp.integrate(func * legandr_polynome, (t, -1, 1)))  # Вычисление Cn
        if abs(cn) < eps:  # Проверка на ноль
            n += 1  # Если ноль, то необходимо вычислить ещё один член ряда
        yp = sp.sympify(cn * legandr_polynome)  # Приведение выражения в понятный для sympy вид
        yp_lambda = sp.lambdify(t, yp, "numpy")     # Конвертирование в лямбда-функцию
                                                            # (что позволяет очень быстро подставлять значения)

        y = yp_lambda(xi)   # Подстановка Xi
        # Если член ряда не содержит переменных, то значение будет одно (скаляр).
        # Тогда необходимо продлить его на весь промежуток.
        if np.isscalar(y):
            y = np.full_like(xi, y)
        plt.plot(xi, y, label=f"{yp}")              # Добавление элемента разложения на график
        ft += yp         # Прибавление элемента к ряду
        i += 1              # Увеличение итератора
    ft_lambda = sp.lambdify(t, ft, "numpy")
    y_ft = ft_lambda(xi)
    return y_ft


def chebyshev(xi, func: sp.core.mul.Mul):
    """ Данная функция раскладывает сигнал по 4 ненулевым членам ряда
    используя полиномы Чебышева.
     Функция отображает каждый элемент ряда на графике.
     Входные параметры: xi - список X, func - функция (в символьном
    виде)
     Выходные параметры: список Y"""

    # Посчитаем элемент c0
    c0 = 1 / pi
    c0 *= sp.N(sp.integrate(func / sp.sympify("sqrt(1-t**2)"), (t, -1, 1)))
    n = 4   # Количество необходимых ненулевых членов ряда
    ft = c0     # Ряд
    i = 1   # Итератор
    eps = pow(10, -7)

    # Аналогично как в функции с полиномами Лежандра
    yp = sp.sympify(ft)
    yp_lambda = sp.lambdify(t, yp)
    y = yp_lambda(xi)
    y = np.full_like(xi, y)
    plt.plot(xi, y, label=f"{ft}")  # Добавление C0 на график
    # Так как функция в варианте является полиномом третьей степени, она раскладывается только на 3 ненулевых члена.

    while i < n:
        chebyshev_polynome = chebyshev_poly(i) # Нахождение полинома Чебышева

        cn = 2 / pi * sp.N(sp.integrate(func * chebyshev_polynome / sp.sympify("sqrt(1-t**2)"), (t, -1, 1)))  # Нахождение Cn
        if abs(cn) < eps:
            continue

        # Аналогичные операции, как и в предыдущем разложении.
        yp = sp.sympify(cn * chebyshev_polynome)
        yp_lambda = sp.lambdify(t, yp, "numpy")
        y = yp_lambda(xi)
        plt.plot(xi, y, label=f"{yp}") # Добавление элемента ряда на график
        ft += yp
        i += 1
    ft_lambda = sp.lambdify(t, ft, "numpy")
    y_ft = ft_lambda(xi)
    return y_ft


def lagger(xi, func):
    """ Данная функция раскладывает сигнал по 4 ненулевым членам ряда
    используя полиномы Лагерра.
     Функция отображает каждый элемент ряда на графике.
     Входные параметры: xi - список X, func - функция (в символьном
    виде)
     Выходные параметры: список Y"""

    n = 4   # Количество требуемых ненулевых членов
    ft = 0  # Ряд
    i = 0   # Итератор
    eps = pow(10, -7)

    def lagger_f(poly):
        """Данная вложенная функция является реализацией функций
        Лаггера
         Входные параметры: poly - полином
         Выходные параметры: e^(-t/2) умноженное на poly"""
        return sp.sympify(f"{e}**(-t/2)") * poly

    while i < n:
        lagger_polynome = lagger_poly(i)    # Нахождение полинома
        cn = sp.N(sp.integrate(func * lagger_f(lagger_polynome), (t, 0, oo))) # Нахождение Cn

        if abs(cn) < eps:
            break

        # Аналогичные операции как и в других разложениях
        yp = sp.sympify(cn * lagger_f(lagger_polynome))
        yp_lambda = sp.lambdify(t, yp, "numpy")
        y = yp_lambda(xi)

        plt.plot(xi, y, label=f"{yp}")  # Добавление элемента на график
        ft += yp
        i += 1
    ft_lambda = sp.lambdify(t, ft, "numpy")
    y_ft = ft_lambda(xi)
    return y_ft


# Запишем входные данные Функции (вариант 1)
legandr_func = sp.sympify(f"0.5*{e}**(0.2*t)-0.2*{e}**(0.2*t)")
chebyshev_func = sp.sympify("3/4*t**3-5/4*t**2+1/4*t")
lagger_func = sp.sympify(f"-0.3*{e}**(-t)+{e}**(-0.3*t)+0.3*{e}**(-0.3 * t)")

# Списки X
x = np.linspace(-1, 1, 1000)  # Для Чебышёва и Лежандра
x2 = np.linspace(0.001, 100, 1000)  # Для Лаггера

# Разложение в ряд Фурье (Лежандр)
series_l = legandr(x, legandr_func)  # Находим список Y для разложения
legandr_func_y = [legandr_func.subs(t, i) for i in x]  # Найдем список Y для оригинальной функции

# Вывод функции, разложения, разложения суммарно.
print("Ряд Фурье(Лежандр, отдельные члены разложения)")
plt.plot(x, legandr_func_y, label="Исходная функция")
plt.legend()
plt.grid()
plt.show()

print("Ряд Фурье(Лежандр, сумма)")
plt.plot(x, legandr_func_y, label="Исходная функция")
plt.plot(x, series_l, label="Весь ряд разложения (сумма)")
plt.legend()
plt.grid()
plt.show()

# Аналогично для других функций

# Разложение в ряд Фурье (Чебышёв)
series_c = chebyshev(x, chebyshev_func)
chebyshev_func_y = [chebyshev_func.subs(t, i) for i in x]
print("Ряд Фурье(Чебышев, отдельные члены разложения)")
plt.plot(x, chebyshev_func_y, label="Исходная функция")
plt.legend()
plt.grid()
plt.show()

print("Ряд Фурье(Чебышев, сумма)")
plt.plot(x, chebyshev_func_y, label="Исходная функция")
plt.plot(x, series_c, label="Весь ряд разложения (сумма)")
plt.legend()
plt.grid()
plt.show()

# Разложение в ряд Фурье (Лаггер)
series_l = lagger(x2, lagger_func)
lagger_func_y = [lagger_func.subs(t, i) for i in x2]
print("Ряд Фурье(Лагерр, отдельные члены разложения)")
plt.plot(x2, lagger_func_y, label="Исходная функция")
plt.legend()
plt.grid()
plt.show()

print("Ряд Фурье(Лаггер, сумма)")
plt.plot(x2, lagger_func_y, label="Исходная функция")
plt.plot(x2, series_l, label="Весь ряд разложения (сумма)")
plt.legend()
plt.grid()
plt.show()
