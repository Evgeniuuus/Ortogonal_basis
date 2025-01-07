import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, e, pi
from sympy.abc import t, theta
from sympy import oo


# Функции для полиномов
def legandr_poly(n: int):
    pn_first_mult = 1 / (2 ** n * factorial(n))
    pn_second_mult = sp.diff(f"(t^2-1)^{n}", t, n)
    return pn_first_mult * pn_second_mult


def chebyshev_poly(n: int):
    tn_first_mult = (-2) ** n * factorial(n) / factorial(2 * n)
    tn_second_mult = sp.sympify("sqrt(1-t**2)")
    tn_third_mult = sp.diff(f"sqrt(1-t**2)**(2*{n}-1)", t, n)
    return sp.simplify(tn_first_mult * tn_second_mult * tn_third_mult)


def lagger_poly(n: int):
    ln_first_mult = sp.sympify(f"{e}**(t)/{factorial(n)}")
    ln_second_mult = sp.diff(sp.sympify(f"t**({n})*{e}**(-t)"), t, n)
    return ln_first_mult * ln_second_mult


def legandr(xi, func: sp.core.mul.Mul):
    n = 4
    ft = 0
    i = 0
    eps = pow(10, -7)
    while i < n:
        legandr_polynome = legandr_poly(i)
        cn = ((2 * i + 1) / 2) * sp.N(sp.integrate(func * legandr_polynome, (t, -1, 1)))
        if abs(cn) < eps:
            n += 1
        yp = sp.sympify(cn * legandr_polynome)
        yp_lambda = sp.lambdify(t, yp, "numpy")
        y = yp_lambda(xi)
        if np.isscalar(y):
            y = np.full_like(xi, y)
        plt.plot(xi, y, label=f"{yp}")
        ft += yp
        i += 1
    ft_lambda = sp.lambdify(t, ft, "numpy")
    y_ft = ft_lambda(xi)
    return y_ft


def chebyshev(xi, func: sp.core.mul.Mul):
    c0 = 1 / pi
    c0 *= sp.N(sp.integrate(func / sp.sympify("sqrt(1-t**2)"), (t, -1, 1)))
    n = 4
    ft = c0
    i = 1
    eps = pow(10, -7)
    yp = sp.sympify(ft)
    yp_lambda = sp.lambdify(t, yp)
    y = yp_lambda(xi)
    y = np.full_like(xi, y)
    plt.plot(xi, y, label=f"{ft}")
    while i < n:
        chebyshev_polynome = chebyshev_poly(i)
        cn = 2 / pi * sp.N(sp.integrate(func * chebyshev_polynome / sp.sympify("sqrt(1-t**2)"), (t, -1, 1)))
        if abs(cn) < eps:
            continue
        yp = sp.sympify(cn * chebyshev_polynome)
        yp_lambda = sp.lambdify(t, yp, "numpy")
        y = yp_lambda(xi)
        plt.plot(xi, y, label=f"{yp}")
        ft += yp
        i += 1
    ft_lambda = sp.lambdify(t, ft, "numpy")
    y_ft = ft_lambda(xi)
    return y_ft


def lagger(xi, func):
    n = 4
    ft = 0
    i = 0
    eps = pow(10, -7)

    def lagger_f(poly):
        return sp.sympify(f"{e}**(-t/2)") * poly

    while i < n:
        lagger_polynome = lagger_poly(i)
        cn = sp.N(sp.integrate(func * lagger_f(lagger_polynome), (t, 0, oo)))

        if abs(cn) < eps:
            break
        yp = sp.sympify(cn * lagger_f(lagger_polynome))

        yp_lambda = sp.lambdify(t, yp, "numpy")
        y = yp_lambda(xi)
        plt.plot(xi, y, label=f"{yp}")
        ft += yp
        i += 1
    ft_lambda = sp.lambdify(t, ft, "numpy")
    y_ft = ft_lambda(xi)
    return y_ft


# Запишем входные данные
legandr_func = sp.sympify(f"0.5*{e}**(0.2*t)-0.2*{e}**(0.2*t)")
chebyshev_func = sp.sympify("3/4*t**3-5/4*t**2+1/4*t")
lagger_func = sp.sympify(f"-0.3*{e}**(-t)+{e}**(-0.3*t)+0.3*{e}**(-0.3 * t)")

x = np.linspace(-1, 1, 1000)  # Для Чебышёва и Лежандра
x2 = np.linspace(0.001, 100, 1000)  # Для Лаггера

# Разложение в ряд Фурье (Лежандр)
series_l = legandr(x, legandr_func)
legandr_func_y = [legandr_func.subs(t, i) for i in x]
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
