from typing import Callable, Union, NamedTuple, Any, Sequence
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from sympy import solve, Symbol
from itertools import combinations, product
import inspect
from tqdm import tqdm
import regex as re
import sympy
np.seterr(all='ignore')


sqrt, cos, sin, tan, pi, acos, asin, atan = (
    sympy.sqrt, sympy.cos, sympy.sin, sympy.tan, sympy.pi, sympy.acos, sympy.asin, sympy.atan)
name_space = {'sqrt' : sqrt, 'cos' : cos, 'sin' : sin, 'tan' : tan, 'pi' : pi,
              'acos' : acos, 'asin' : asin, 'atan' : atan}


def double_integral(function: Callable,
                    limits: list[str],
                    precision: int = 300,
                    plot_on: bool = True,
                    manual_boundaries: dict = None):

    """
    Designed for double integration of relatively simple analytical functions. Function (integrand) might be provided
    as a Callable object.

    Limits (inequalities) are parsed and analysed to obtain the vertices of the domain. The coordinates of these
    vertices serve as a guidance for a rectangular field, which is expected to completely overlap with the domain.
    After the 'envelop' rectangle is found, a meshgrid of points is checked one by one to create a mask of points,
    which belong to the domain.

    Notice: in some cases like e.g. x ** 2 + y ** 2 < 4, a single inequality defines the finite domain, although
    there are no vertices to compute. For such a situation, it is clearly required to pass manual_boundaries
    e.g. dict(x=(-2, 2), y=(-2, 2)).

    Integrand is evaluated at the sample points from the domain and multiplied by the 'column area', which depends
    on precision. (In fact, Simpson's rule is used). Also, upper sum and lower sum are computed to estimate the
    maximum possible error of integration.

    If plotting option is on, 3 surfaces are plotted: 0-surface (just a plain at z = 0), integrated volume,
    overall function (which covers the domain and extends further to show the nearby function behaviour). In case of
    automatically found domain vertices, constraints will be plotted as dashed lines at z=0 plain).

    Parameters:
        function (Callable): a valid Python function is expected. It should return int or float for correct output.
        limits (list): a list of strings. Each string is an inequality (can be strict, double etc.). It is important
            that variables in these inequalities are called 'x' and 'y'. If there is a single inequality, no
            automatic vertices can be derived and manual_boundaries are required.
        precision (int): number of sampled points along a single axis. The computational complexity increases as
            a square of precision parameter. Defaults to 300.
        plot_on (bool): allows to plot the integral. Defaults to True.
        manual_boundaries (dict): a dictionary with only 3 possible keys accepted: 'x', 'y', 'z'.
            'x' value stands for x-bounds which are expected to completely encompass the domain;
            'y' value stands for y-bounds which are expected to completely encompass the domain;
            'z' value stands for z-bounds which only influence plt plotting limits;
            keys can be passed independently. Defaults to None.

    """

    # Assertions
    assert isinstance(function, Callable), 'functions must be a Callable object'
    assert isinstance(limits, list), 'limits must be a list'
    assert len(limits) >= 1, 'There must be at least 1 inequality to define a finite domain'
    for lim in limits:
        assert isinstance(lim, str), 'Each limit must be a string'
    assert isinstance(plot_on, bool), 'plot_on must be boolean'
    assert isinstance(precision, int) and precision > 9, 'precision must an integer larger than 9'
    x_boundaries, y_boundaries, z_boundaries = None, None, None
    if manual_boundaries:
        assert isinstance(manual_boundaries, dict), 'manual_boundaries must be a dict if provided'
        assert 1 <= len(manual_boundaries) <= 3, 'manual_boundaries can have 1, 2 or 3 tuples of boundaries'
        for axis, b in manual_boundaries.items():
            assert axis in ['x', 'y', 'z'], f'Unknown axis {axis}, can only be x, y, z'
            assert isinstance(b, Sequence) and len(b) == 2, 'each boundary must be a 2-tuple'
            assert isinstance(b[0], Union[int, float]) and isinstance(b[1], Union[int, float]), f'Invalid boundary: {b}'
            assert b[1] > b[0], 'upper boundary must larger than lower boundary'
        x_boundaries = manual_boundaries.get('x')
        y_boundaries = manual_boundaries.get('y')
        z_boundaries = manual_boundaries.get('z')


    # Parse limits:
    x, y = Symbol('x', real=True), Symbol('y', real=True)
    unit_limits = []
    for limit in limits:
        limit = limit.replace('<=', '<').replace('>=', '>')
        assert '=' not in limit, f'Invalid limit: {limit}'
        split_limit = re.findall("(?<=^|<|>)([^<>]+)([<>])([^<>]+)(?=$|<|>)", str(limit), overlapped=True)
        for unit in split_limit:
            if unit[1] == '<':
                unit_limits.append((unit[0].replace(' ', ''), unit[2].replace(' ', '')))
            else:
                unit_limits.append((unit[2].replace(' ', ''), unit[0].replace(' ', '')))

    class Equation(NamedTuple):
        """
        A helper class for neet representation of constraints.
        """
        x_is_equal_to: Any
        x_is_of_type: str
        y_is_equal_to: Any
        y_is_of_type: str

        def __str__(self):
            return ('x = ' + str(self.x_is_equal_to) + f'({self.x_is_of_type})\n' +
                    'y = ' + str(self.y_is_equal_to) + f'({self.y_is_of_type})\n')

    def solve_x_y(function_pair: tuple, single_var: bool = False):
        """
        Helper function to find the intersection of 2 functions.
        If single_var is False, sympy returns a general symbolic solution.
        If single_var is True, the result is a number.
        """
        sol = (solve(eval(function_pair[0] + ' - (' + function_pair[1] + ')', {}, {'x': x, 'y': y, **name_space}), x),
                solve(eval(function_pair[0] + ' - (' + function_pair[1] + ')', {}, {'x': x, 'y': y, **name_space}), y))
        if single_var:
            sol = list(filter(lambda t: t, sol))
        return sol

    def check_conditions(p: tuple, strict: bool = False, pb: tqdm = None):
        """
        Helper function to check if a point 'p' meet all the inequalities (strictly if 'strict' is True).
        Also, can update a provided tqdm progressbar 'pb'.
        """
        if pb:
            pb.update(1)
        for check in unit_limits:
            try:
                left_part = eval(check[0], {}, {'x': float(p[0]), 'y': float(p[1]), **name_space})
                right_part = eval(check[1], {}, {'x': float(p[0]), 'y': float(p[1]), **name_space})
            except ZeroDivisionError:
                return False
            if isinstance(left_part, complex) or isinstance(right_part, complex):
                return False
            if strict:
                if left_part >= right_part:
                    return False
            else:
                if left_part > right_part:
                    return False
        return True

    # Found overall boundaries
    limit_counter = []

    # If not every boundary is set manually. Expressing every condition using both variables (if possible) and
    # assigning corresponding tags
    if not (x_boundaries and y_boundaries):
        eqs_x, eqs_y = [], []
        for limit in unit_limits:
            eq_x, eq_y = solve_x_y(limit)
            if len(eq_x) <= 1 and len(eq_y) <= 1:
                limit_counter.append(unit_limits.index(limit))
                eqs_x.append(eq_x)
                eqs_y.append(eq_y)
            else:
                for x_, y_ in list(product(list(eq_x), list(eq_y))):
                    limit_counter.append(unit_limits.index(limit))
                    eqs_x.append([x_])
                    eqs_y.append([y_])

        for i, eq in enumerate(eqs_x):
            if not eq:
                type_ = 'empty'
            elif str(eq[0]).isnumeric():
                type_ = 'numeric'
            else:
                type_ = 'function'
            eqs_x[i] = [eq[0] if eq else None, type_]
        for i, eq in enumerate(eqs_y):
            if not eq:
                type_ = 'empty'
            elif str(eq[0]).isnumeric():
                type_ = 'numeric'
            else:
                type_ = 'function'
            eqs_y[i] = [eq[0] if eq else None, type_]
        eqs = [Equation(x_is_equal_to=a[0], y_is_equal_to=b[0], x_is_of_type=a[1], y_is_of_type=b[1])
               for a, b in zip(eqs_x, eqs_y)]

        # Description of constraints:
        max_length = max([max(len(a), len(b)) for (a, b) in unit_limits]) + 3
        max_length_x = max([len(str(q.x_is_equal_to)) for q in eqs]) + 3
        max_length_y = max([len(str(q.y_is_equal_to)) for q in eqs]) + 3
        print(f'Precision = {precision}'.center(max_length * 2 + 6) + '|' + 'Boundaries:'.center(23 + max_length_x + max_length_y))
        print('-' * (max_length * 2 + max_length_y + max_length_x + 31))
        print('Limits received:'.center(max_length * 2 + 6) + '|' + 'X ='.center(max_length_x) +
              '|' + 'X type'.center(10) + '|' + 'Y ='.center(max_length_y) + '|' + 'Y type'.center(10))
        print('-' * (max_length * 2 + max_length_y + max_length_x + 31))
        print(*[f'{l + 1}) {unit_limits[l][0].rjust(max_length)} < {unit_limits[l][1].ljust(max_length)}'
                f'|{str(q.x_is_equal_to).center(max_length_x)}'
                f'|{str(q.x_is_of_type).center(10)}'
                f'|{str(q.y_is_equal_to).center(max_length_y)}'
                f'|{str(q.y_is_of_type).center(10)}'
                for (l, q) in zip(limit_counter, eqs)], sep='\n')
        print('-' * (max_length * 2 + max_length_y + max_length_x + 31))

        # Searching for vertices (points where 2 conditions intersect)
        points = []
        for eq_1, eq_2 in combinations(eqs, 2):
            x_point, y_point = None, None
            if eq_1.x_is_of_type == 'numeric' and eq_2.x_is_of_type == 'numeric':
                continue # parallel lines
            if eq_1.y_is_of_type == 'numeric' and eq_2.y_is_of_type == 'numeric':
                continue # parallel lines
            if eq_1.x_is_of_type == 'numeric' or eq_2.x_is_of_type == 'numeric':
                x_point = eq_1.x_is_equal_to if eq_1.x_is_of_type == 'numeric' else eq_2.x_is_equal_to
            if eq_1.y_is_of_type == 'numeric' or eq_2.y_is_of_type == 'numeric':
                y_point = eq_1.y_is_equal_to if eq_1.y_is_of_type == 'numeric' else eq_2.y_is_equal_to

            if x_point is not None and y_point is not None:
                points.append((x_point, y_point))
                continue # intersection of axis-aligned constraints
            elif x_point is not None:  # plugging in x
                if eq_1.y_is_of_type == 'function':
                    y_point = eval(str(eq_1.y_is_equal_to), {}, {'x' : x_point, **name_space})
                    points.append((x_point, y_point))
                elif eq_2.y_is_of_type == 'function':
                    y_point = eval(str(eq_2.y_is_equal_to), {}, {'x': x_point, **name_space})
                    points.append((x_point, y_point))
                else:
                    continue
            elif y_point is not None:  # plugging in y
                if eq_1.x_is_of_type == 'function':
                    x_point = eval(str(eq_1.x_is_equal_to), {}, {'y' : y_point, **name_space})
                    points.append((x_point, y_point))
                elif eq_2.x_is_of_type == 'function':
                    y_point = eval(str(eq_2.x_is_equal_to), {}, {'y': y_point, **name_space})
                    points.append((x_point, y_point))
                else:
                    continue
            elif eq_1.x_is_of_type == 'function' and eq_2.x_is_of_type == 'function':
                y_point = solve_x_y((str(eq_1.x_is_equal_to), str(eq_2.x_is_equal_to)), single_var=True)
                if not y_point:
                    continue
                y_point = y_point[0]
                for y_point_ in y_point:
                    try:
                        x_point = eval(str(eq_1.x_is_equal_to), {}, {'y' : y_point_, **name_space})
                    except TypeError:
                        x_point = eval(str(eq_2.x_is_equal_to), {}, {'y': y_point_, **name_space})
                    points.append((x_point, y_point_))
            elif eq_1.y_is_of_type == 'function' and eq_2.y_is_of_type == 'function':
                x_point = solve_x_y((str(eq_1.y_is_equal_to), str(eq_2.y_is_equal_to)), single_var=True)
                if not x_point:
                    continue
                x_point = x_point[0]
                for x_point_ in x_point:
                    try:
                        y_point = eval(str(eq_1.y_is_equal_to), {}, {'x' : x_point_, **name_space})
                    except TypeError:
                        y_point = eval(str(eq_2.y_is_equal_to), {}, {'x': x_point_, **name_space})
                    points.append((x_point_, y_point))
            else:
                continue

        # Removing intersection outside the domain and finding boundaries (if not provided explicitly)
        points = list(set(points))
        points = list(filter(lambda t: t[0] is not None and t[1] is not None, points))
        points = list(filter(lambda t: check_conditions(t, strict=False), points))
        min_x, max_x = np.min([t[0] for t in points]), np.max([t[0] for t in points])
        if x_boundaries:
            min_x, max_x = x_boundaries
        min_y, max_y = np.min([t[1] for t in points]), np.max([t[1] for t in points])
        if y_boundaries:
            min_y, max_y = y_boundaries
        if len(points) < 3:
            raise ValueError('Not enough vertices found. Either the domain is empty or consider providing the manual_boundaries.')
        print(f'Vertices of the domain are: {points}')

    else:
        min_x, max_x = x_boundaries
        min_y, max_y = y_boundaries

    # Widening the 'envelop' of the domain, computing function values in grid nodes
    delta_x = (float(max_x) - float(min_x)) * 0.3
    delta_y = (float(max_y) - float(min_y)) * 0.3
    delta_x_mini = (float(max_x) - float(min_x)) * 0.01
    delta_y_mini = (float(max_y) - float(min_y)) * 0.01

    # Compute the integral using Simpson's method:
    X, Y = np.meshgrid(np.linspace(float(min_x) - delta_x_mini, float(max_x) + delta_x_mini, int(precision * 1.02), endpoint=False),
                        np.linspace(float(min_y)  - delta_y_mini, float(max_y) + delta_y_mini, int(precision * 1.02), endpoint=False))
    step = np.abs(float(min_y) - float(max_y)) / (2 * precision)

    def f_wrapped(a, b):
        try:
            v = function(a, b)
            if isinstance(v, float) or isinstance(v, int):
                return float(v)
            else:
                return np.inf
        except:
            return np.inf

    V = np.vectorize(f_wrapped)(X, Y)
    read_bar_format = "%s{l_bar}%s{bar}%s{r_bar}" % ("\033[1;30m", "\033[1;30m", "\033[1;30m")

    # The bottleneck part - mask creating
    with tqdm(total=len(V) ** 2 + 1, ascii=" #", desc='Computing the integral... ', unit=' points checked',
              ncols=100, initial=0, bar_format=read_bar_format) as pb:
        mask = ~np.vectorize(lambda a, b: True if check_conditions((a, b), strict=False, pb=pb) else False)(X, Y)
        pb.close()

    # Simpson's rule, lower sum, upper sum:
    V[mask] = 0
    V_1 = np.vectorize(lambda a, b: f_wrapped(a + step, b))(X, Y)
    V_2 = np.vectorize(lambda a, b: f_wrapped(a, b + step))(X, Y)
    V_3 = np.vectorize(lambda a, b: f_wrapped(a - step, b))(X, Y)
    V_4 = np.vectorize(lambda a, b: f_wrapped(a, b - step))(X, Y)
    V_5 = np.vectorize(lambda a, b: f_wrapped(a - step, b - step))(X, Y)
    V_6 = np.vectorize(lambda a, b: f_wrapped(a - step, b + step))(X, Y)
    V_7 = np.vectorize(lambda a, b: f_wrapped(a + step, b - step))(X, Y)
    V_8 = np.vectorize(lambda a, b: f_wrapped(a + step, b + step))(X, Y)
    V_1[mask], V_2[mask], V_3[mask], V_4[mask], V_5[mask], V_6[mask], V_7[mask], V_8[mask] = [0] * 8
    (V_1[V_1 == np.nan], V_2[V_2 == np.nan], V_3[V_3 == np.nan], V_4[V_4 == np.nan], V_5[V_5 == np.nan],
     V_6[V_6 == np.nan], V_7[V_7 == np.nan], V_8[V_8 == np.nan]) = [0] * 8
    column_area = (float(max_x) - float(min_x)) * (float(max_y) - float(min_y)) / (precision ** 2)
    volume = np.nansum((16 * V + 4 * (V_1 + V_2 + V_3 + V_4) + (V_5 + V_6 + V_7 + V_8)) * column_area / 36)
    volume_abs = np.nansum((16 * np.abs(V) + 4 * (np.abs(V_1) + np.abs(V_2) + np.abs(V_3) + np.abs(V_4)) +
                            (np.abs(V_5) + np.abs(V_6) + np.abs(V_7) + np.abs(V_8))) * column_area / 36)
    volume_max = np.nansum((np.max(np.array([V, V_1, V_2, V_3, V_4, V_5, V_6, V_7, V_8]), axis=0)) * column_area)
    volume_min = np.nansum((np.min(np.array([V, V_1, V_2, V_3, V_4, V_5, V_6, V_7, V_8]), axis=0)) * column_area)
    error = 0.5 * (volume_max - volume_min)
    print('Integral value: '.rjust(28), f'{volume:.6g}')
    if volume_abs != volume:
        print('Total volume: '.rjust(28), f'{volume_abs:.6g}')
    print('Min: '.rjust(28), f'{volume_min:.6g}')
    print('Max: '.rjust(28), f'{volume_max:.6g}')

    # Plotting:
    if plot_on:
        max_v, min_v = np.max(V), np.min(V)
        max_v, min_v = max(max_v + (max_v - min_v) * 0.5, 0), min(min_v - (max_v - min_v) * 0.5, 0)
        X_wide, Y_wide = np.meshgrid(np.linspace(float(min_x) - delta_x, float(max_x) + delta_x, 100),
                                     np.linspace(float(min_y) - delta_y, float(max_y) + delta_y, 100))
        V_wide = np.vectorize(f_wrapped)(X_wide, Y_wide)
        try:
            V_wide[V_wide > max_v] = np.inf
            V_wide[V_wide < min_v] = np.inf
        except ValueError:
            pass
        V_wide = np.nan_to_num(V_wide, True, 0)
        V_wide = V_wide * 0.97
        V_wide_baseline = np.vectorize(lambda a, b: 0.0)(X_wide, Y_wide)
        coolwarm0, coolwarm1 = colormaps['viridis'], colormaps['Blues']
        new_cmap0, new_cmap1 = coolwarm0(np.linspace(0, 1, 256)), coolwarm1(np.linspace(0, 1, 256))[::-1]
        cmap_under_0 = new_cmap0[:len(new_cmap0) // 2]
        cmap_over_0 = new_cmap0[len(new_cmap0) // 2:]
        cmap_over_0[0, -1], cmap_over_0[1:, -1], cmap_over_0[-1, -1] = 0, 1, 1
        cmap_under_0[0, -1], cmap_under_0[1:, -1], cmap_under_0[-1, -1] = 0, 1, 0
        new_cmap1[0, -1], new_cmap1[1:, -1], new_cmap1[-1, -1] = 0.3, 0.3, 0.3
        new_cmap0, new_cmap1 = ListedColormap(new_cmap0), ListedColormap(new_cmap1)
        cmap_over_0, cmap_under_0 = ListedColormap(cmap_over_0), ListedColormap(cmap_under_0)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X_wide, Y_wide, V_wide_baseline, color='gray', linewidth=0, antialiased=False,
                        alpha=0.25, zorder=0)
        ax.plot_surface(X_wide, Y_wide, V_wide, cmap=new_cmap1, linewidth=1, antialiased=False, zorder=1)
        V_over_0, V_under_0 = V.copy(), V.copy()
        V_over_0[V_over_0 <= 0] = 0
        V_under_0[V_under_0 >= 0] = 0
        ax.plot_surface(X, Y, V_over_0, cmap=cmap_over_0, linewidth=0, antialiased=False, zorder=2)
        ax.plot_surface(X, Y, V_under_0, cmap=cmap_under_0, linewidth=0, antialiased=False, zorder=4)
        ax.set_xlim([float(min_x) - delta_x, float(max_x) + delta_x])
        ax.set_ylim([float(min_y) - delta_y, float(max_y) + delta_y])
        if z_boundaries:
            ax.set_zlim(z_boundaries)
        else:
            ax.set_zlim([min_v, max_v])
        function_code = ''.join(inspect.getsource(function).split(':')[1:]).replace('\n', ' ').split(',')[0]
        if 'return' in function_code:
            function_code = sorted([f for f in function_code.split('return')], key=lambda x: len(x.replace(' ', '')))[-1]
            function_code = function_code.replace('*', '\\text{*}')
        if len(function_code) > 50:
            function_code = 'f(x, y)'
        title = '$\\' + 'int \\' + f'int {function_code} dydx$'
        title += ' | ' + ' | '.join(limits) + f'\nvalue = {volume:.4g} $' + '\\' + f'pm$ {error:.4g}'
        title += f'{"\n" if volume_abs == volume else "| volume = " + f"{volume_abs:.4g} $units^{{3}}$\n"}'
        title += f'precision: {1/column_area:.0f} per $unit^{{2}}$'
        ax.set_title(title)


        # Plotting boundaries
        if not (x_boundaries and y_boundaries):
            for eq in eqs:
                if eq.x_is_of_type == 'numeric':
                    ax.plot3D([eq.x_is_equal_to] * precision,
                              np.linspace(float(min_y) - delta_y, float(max_y) + delta_y, precision), [0],
                              linewidth=2, color='black', zorder=3, linestyle='dotted')
                if eq.y_is_of_type == 'numeric':
                    ax.plot3D(np.linspace(float(min_x) - delta_x, float(max_x) + delta_x, precision),
                              [eq.y_is_equal_to] * precision, [0], linewidth=3, color='black', zorder=2, linestyle='dotted')
                if eq.y_is_of_type == 'function':
                    def curve(a: np.array):
                        x_vals, y_vals = [], []
                        for a_ in a:
                            value = eval(str(eq.y_is_equal_to), {}, {'x' : a_, **name_space})
                            try:
                                value = float(value)
                                x_vals.append(a_)
                                y_vals.append(value)
                            except TypeError:
                                continue
                        return x_vals, y_vals
                    x_values, y_values = curve(np.linspace(float(min_x) - delta_x, float(max_x) + delta_x, precision))
                    ax.plot3D(x_values, y_values, [0], linewidth=2, color='black', zorder=3, linestyle='dotted')
                if eq.x_is_of_type == 'function':
                    def curve(a: np.array):
                        x_vals, y_vals = [], []
                        for a_ in a:
                            value = eval(str(eq.x_is_equal_to), {}, {'y' : a_, **name_space})
                            try:
                                value = float(value)
                                y_vals.append(a_)
                                x_vals.append(value)
                            except TypeError:
                                continue
                        return x_vals, y_vals
                    x_values, y_values = curve(np.linspace(float(min_y) - delta_y, float(max_y) + delta_y, precision))
                    ax.plot3D(x_values, y_values, [0], linewidth=2, color='black', zorder=3, linestyle='dotted')
        plt.show()

    return volume

if __name__ == '__main__':

    double_integral(lambda x, y: 1 / (3 * x + 2 * y), ['x < y < 5 * x', '2 < x + 2 * y < 4'],
                    plot_on=True, precision=300)

    double_integral(lambda x, y: 1, ['x ** 2 + y ** 2 - 2 * (x ** 2 + y ** 2) ** 0.5 < -x'],
                      plot_on=True, manual_boundaries=dict(x=(-3.5, 3.5), y=(-3.5, 3.5)), precision=300)

    def function(x, y):
        return 1 if x > 0.5 and y > 0.5 else 0.5

    double_integral(function, ['0 < x', '0 < y', 'y < x + 1', 'y > x - 1', 'y < 2 - x'],
                    plot_on=True, precision=700)

    double_integral(lambda x, y: (4 - x ** 2 - y ** 2) ** 0.5 + 1,
                ['0 < x ** 2 + y ** 2 < 2 * x', '- pi/2 < atan(y/x) < pi/2'],
                       plot_on=True, precision=1000, manual_boundaries=dict(x=(-2.1, 2.1), y=(-1, 2.1)))

