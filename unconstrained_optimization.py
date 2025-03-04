from typing import Union, Sequence, Callable
from numbers import Number
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
import sympy
import inspect
import copy
import time
import datetime


class Task:
    """
    class Task represents the function and corresponding attributes
    """

    def __init__(self, expression, n_var: int = None, x0: Union[Sequence | Number | np.ndarray] = None,
                 x_min: Sequence[Number | Sequence | np.ndarray] = None,
                 external_grad: Callable = None, external_hess: Callable = None):
        """
        :param expression: string-like function, each variable must be written as 'x[i]' or 'x_i' or 'xi'
        :param n_var: total number of variables in the function
        :param x0: starting point
        :param x_min: list of known minimizers
        ..............................................................................................................\n
        **How to provide expression?**\n
        With a string, where 'x' is the variable: 'x ** 2 + 2' / 'x[0] ** 3 - x[1]' / 'x0 * x1 * x2 - x3 * x4' / '(x_1 + x_2) ** 0.5' etc.
            > Make sure that not-built-in functions are in sympy form, e.g. 'sympy.cos(x0)'\n
            > Do not use other variable names like 'y' or 'z'\n
            > Make sure that function yield a scalar value (not a sequence)\n
        With a function object (both lambda and def expressions are supported)
            > Argument can be a single array, e.g. lambda x: x[0] ** 2 - x[1]\n
            > Argument can be a single scalar for 1D optimization: lambda x: math.sin(x ** 4 -1)\n
            > Several positional arguments possible: lambda x1, x2: x1 * x2 - 10\n
            > If function has other arguments except for the optimized ones, make them keyword args with default value, e.g:
                def my_f(x1, x2, x3, parameter=1):
                    return x1 ** parameter - x2 * x3\n\n
        ..............................................................................................................\n
        **How to provide n_var?**\n
            Just enter a single integer with the presumed number of variables.\n\n
        ..............................................................................................................\n
        **How to provide initial point?**\n
            It can be omitted. Then it will be set to 0-array by default.\n
        It can be provided as a single sequence with the same number of elements as n_var has: n_var=2, x0=[1, 1]\n\n
        ..............................................................................................................\n
        **How to provide minimizers?**\n
            It can be omitted.\n
        It can be provided as a sequence of length n_var (if a single minimizer) or as a sequence of sequences each of
        length n_var (several minimizers). x_min=[[1, 1], [2, 2]]\n
        In case of 1-variable task, it is possible to provide a single scalar for one minimizer of a sequence of
        scalars for multiple minimizers. x_min=5.2 / x_min=(-2, 1, 4)\n\n
        ..............................................................................................................\n
        **How to provide external gradient?**\n
        It can be omitted.
            > If main function is sympy expression: sympy tools will be used for auto-gradient.\n
            > If main function is a function object: numerical tools will be used (with precision ~1e-6)\n
        It can be provided as an explicit function (both lambda and def are possible)
            > There the same rules as for main function (if several positional args -> they would be converted to one
            array by wrapping function)\n
            > Output of gradient function must be an array of length n_var.\n\n
        ..............................................................................................................\n
        **How to provide external hessian?**\n
        It can be omitted.
            > If main function is sympy expression: sympy tools will be used for auto-hessian.\n
            > If main function is a function object: numerical tools will be used (with precision ~1e-4)\n
        It can be provided as an explicit function (both lambda and def are possible)
            > There the same rules as for main function (if several positional args -> they would be converted to
            one array by wrapping function)\n
            > Output of hessian function must be an array of shape (n_var, n_var).

        """
        # Basic assertions:
        assert n_var or x0, 'Either n_var or x0 (or both) must be provided'
        if n_var:
            assert isinstance(n_var, int) and n_var > 0, f'n_var {n_var} is not a positive integer'
        assert isinstance(expression, Union[str, Callable]), 'expression must be string or Callable'
        assert isinstance(external_grad, Union[None, Callable]), 'external_grad must be Callable'
        assert isinstance(external_hess, Union[None, Callable]), 'external_hess must be Callable'
        assert isinstance(x_min, Union[None, Sequence, Number, np.ndarray]), 'invalid x_min type'

        # x0 preprocessing
        assert isinstance(x0, Union[None, Sequence, Number, np.ndarray]), 'invalid x0 type'
        if x0 is None:
            x0 = np.zeros(n_var, dtype=np.float32)
        elif isinstance(x0, Sequence | np.ndarray):
            x0 = np.array(x0, dtype=np.float32).flatten()
            n_var = len(x0)
        else:
            x0 = np.array([x0], dtype=np.float32)
            n_var = len(x0)

        # x_min preprocessing
        if isinstance(x_min, Sequence | np.ndarray):
            try:
                x_min = np.array(x_min, dtype=np.float32)
            except ValueError:
                raise AssertionError('Minimizers are of different shape')
            if len(x_min.shape) == 1:
                if n_var == 1:
                    x_min = np.array(x_min, dtype=np.float32).reshape(-1, 1)
                else:
                    x_min = np.array([x_min], dtype=np.float32)
        if not isinstance(x_min, Sequence | np.ndarray) and (x_min is not None):
            x_min = np.array([[x_min]], dtype=np.float32)

        # expression:
        if isinstance(expression, str):
            expression = expression.replace('^', '**')
            wrong_pattern = re.findall(r'x_[0-9]', expression)
            wrong_pattern += re.findall(r'x[0-9]', expression)
            for case in wrong_pattern:
                expression = expression.replace(case, f'x[{re.findall(r"[0-9]+", case)[0]}]')
        if isinstance(expression, Callable):
            args = len(inspect.getfullargspec(expression).args)
            kwargs = inspect.getfullargspec(expression).defaults
            kwargs = 0 if kwargs is None else len(kwargs)
            n_args = args - kwargs
            if n_args > 1:
                expr = copy.copy(expression)
                expression = lambda x: float(expr(*x))

        # external_grad
        if external_grad is None:
            n_args = 1
        else:
            args = len(inspect.getfullargspec(external_grad).args)
            kwargs = inspect.getfullargspec(external_grad).defaults
            kwargs = 0 if kwargs is None else len(kwargs)
            n_args = args - kwargs
        if n_args > 1:
            ex_grad = copy.copy(external_grad)
            external_grad = lambda x: ex_grad(*x)
        # external_hess
        if external_hess is None:
            n_args = 1
        else:
            args = len(inspect.getfullargspec(external_hess).args)
            kwargs = inspect.getfullargspec(external_hess).defaults
            kwargs = 0 if kwargs is None else len(kwargs)
            n_args = args - kwargs
        if n_args > 1:
            ex_hess = copy.copy(external_hess)
            external_hess = lambda x: ex_hess(*x)

        # Further assertions:
        assert len(x0) == n_var, f'x0 has {len(x0)} components, {n_var} were expected'
        if x_min is not None:
            for minimizer in x_min:
                assert len(minimizer) == n_var, f'some minimizers have {len(minimizer)}D shape, {n_var}D was expected'
        if isinstance(expression, Callable):
            try:
                check = expression(x0)
                if isinstance(check, Sequence | np.ndarray) and len(check) == 1:
                    check = check[0]
            except BaseException:
                raise AssertionError('Provided function does not work as expected')
            assert isinstance(check, Number), f'function output must be a scalar, not {type(check)}'
        if isinstance(expression, str):
            set_of_vars = set([int(x[1:-1]) for x in re.findall(r'\[.\]|\[..\]', expression)])
            if len(set_of_vars) == 0:
                expression = expression.replace('x', f'x[0]')
            for i, var in enumerate(set_of_vars):
                expression = expression.replace(f'[{var}]', f'[{i}]')
            num = max([int(x[1:-1]) for x in re.findall(r'\[.\]|\[..\]', expression)]) + 1
            assert n_var == num, f'number of variables does not match {num} != {n_var}'
            x = x0
            try:
                check = float(eval(expression))
            except NameError as ne:
                wrong_f = str(ne).split("'")[1]
                raise AssertionError(f'function does not work as expected - {str(ne)}, try "sympy.{wrong_f}" instead')
            assert isinstance(check, Number), f'function output must be a scalar, not {type(check)}'
        if isinstance(external_grad, Callable):
            try:
                grad_check = external_grad(x0) if isinstance(external_grad(x0), Sequence | np.ndarray) else [
                    external_grad(x0)]
                grad_check = np.array(grad_check, dtype=np.float32)
            except BaseException:
                raise AssertionError('Provided Callable gradient does not suit')
            assert len(grad_check) == n_var, f'External_grad yields array of len {len(grad_check)}, {n_var} expected'
        if isinstance(external_hess, Callable):
            try:
                hess_check = external_hess(x0) if isinstance(external_hess(x0), Sequence | np.ndarray) else [
                    external_hess(x0)]
                hess_check = np.array(hess_check, dtype=np.float32)
            except BaseException:
                raise AssertionError('Provided Callable hessian does not suit')
            assert hess_check.shape == (n_var, n_var), (f'External_hess yields array of shape {hess_check.shape}, '
                                                        f'{(n_var, n_var)} expected')

        self.expression = expression
        self.__n_var = n_var
        self.x0 = x0
        self.min = x_min
        self.external_grad = external_grad
        self.external_hess = external_hess
        if isinstance(expression, str):
            x = [sympy.symbols(f'x_{i}') for i in range(self.__n_var)]
            f = eval(self.expression)
            self.__gradient = [f.diff(x[i]) for i in range(self.__n_var)]
            self.__hessian = [[f.diff(x[i]).diff(x[j]) for j in range(self.__n_var)] for i in range(self.__n_var)]
        return

    @property
    def n_var(self):
        return self.__n_var

    def __repr__(self):
        task_repr = ' TASK '.center(40, '-') + '\n'
        if isinstance(self.expression, str):
            task_repr += 'Function (string):'.ljust(30) + f'{self.expression}\n'
        else:
            task_repr += 'Function (Callable object):'.ljust(30) + f'{self.expression}\n'
        task_repr += 'Number of variables:'.ljust(30) + f'{self.__n_var}\n'
        if self.x0 is not None:
            task_repr += 'Initial point:'.ljust(30) + f'{self.x0}\n'
        else:
            task_repr += 'Initial point:'.ljust(30) + 'not set\n'
        if self.min is not None:
            task_repr += 'True optimizers:'.ljust(30) + f'{[m.tolist() for m in self.min]}\n'
        else:
            task_repr += 'True optimizers:'.ljust(30) + 'not provided\n'
        if self.external_grad is None:
            task_repr += 'Gradient:'.ljust(30) + 'automatically\n'
        else:
            task_repr += 'Gradient (external function):'.ljust(30) + f'{self.external_grad}\n'
        if self.external_hess is None:
            task_repr += 'Hessian:'.ljust(30) + 'automatically\n'
        else:
            task_repr += 'Hessian (external function):'.ljust(30) + f'{self.external_hess}\n'
        task_repr += '-' * 40
        return task_repr

    def __str__(self):
        return self.__repr__()

    def __value(self, point: Sequence) -> float:
        """
        Yields function value at x
        :param point: coordinates
        :return:
        """
        if isinstance(point, Number):
            point = np.array([point], dtype=np.float64)
        assert len(point) == self.__n_var, f'point must have {self.__n_var} components, not {len(point)}'
        x = np.array(point, dtype=np.float64)
        if isinstance(self.expression, Callable):
            output = self.expression(x)
            return output if isinstance(output, Number) else output[0]
        return float(eval(self.expression))

    def value(self, point: Sequence) -> np.float32:
        """
        Yields function value at x
        :param point: coordinates
        :return:
        """
        return np.float64(self.__value(point))

    def grad(self, point: Sequence) -> np.ndarray:
        """
        Yields vector-gradient at a point 'point'
        :param point: coordinates
        :return:
        """
        if isinstance(point, Number):
            point = np.array([point], dtype=np.float64)
        point = np.array(point, dtype=np.float64)
        assert len(point) == self.__n_var, f'point must have {self.__n_var} components'
        if self.external_grad:
            gradient_ = self.external_grad(point)
            gradient_ = gradient_ if isinstance(gradient_, Sequence | np.ndarray) else [gradient_]
            return np.array(gradient_, dtype=np.float64)
        if isinstance(self.expression, str):
            subs = dict([(f'x_{i}', point[i]) for i in range(self.__n_var)])
            gradient_eval = [self.__gradient[i].evalf(subs=subs) for i in range(self.__n_var)]
            return np.array(gradient_eval, dtype=np.float64)
        if isinstance(self.expression, Callable):
            delta: np.float64 = 0.5 * 10 ** (-8)
            gradient_ = np.zeros(self.__n_var, dtype=np.float64)
            for i in range(self.__n_var):
                direction = np.zeros(self.__n_var, dtype=np.float64)
                direction[i] = delta
                gradient_[i] = (self.__value(point + direction) - self.__value(point - direction)) / (2 * delta)
            return np.array(gradient_, dtype=np.float64)

    def hess(self, point: Sequence) -> np.ndarray:
        """
        Yields a Hessian matrix at a point 'point'
        :param point: coordinates
        :return:
        """
        if isinstance(point, Number):
            point = np.array([point], dtype=np.float64)
        point = np.array(point, dtype=np.float64)
        assert len(point) == self.__n_var, f'point must have {self.__n_var} components'
        if self.external_hess:
            return np.array(self.external_hess(point), dtype=np.float64)
        if isinstance(self.expression, str):
            subs = dict([(f'x_{i}', point[i]) for i in range(self.__n_var)])
            hessian_eval = [[self.__hessian[i][j].evalf(subs=subs) for i in range(self.__n_var)] for j in
                            range(self.__n_var)]
            return np.array(hessian_eval, dtype=np.float64)
        if isinstance(self.expression, Callable):
            delta, epsilon = 2 * 10 ** (-4), 2 * 10 ** (-4)
            hessian_ = np.zeros((self.__n_var, self.__n_var), dtype=np.float64)
            for i in range(self.__n_var):
                for j in range(self.__n_var):
                    direction1, direction2 = np.zeros(self.__n_var, dtype=np.float64), np.zeros(self.__n_var,
                                                                                                dtype=np.float64)
                    direction1[i] += delta
                    direction2[j] += epsilon
                    hessian_[i, j] += ((self.__value(point + direction1 + direction2) - self.__value(
                        point - direction1 + direction2)) / (2 * delta) - ((self.__value(
                        point + direction1 - direction2) - self.__value(
                        point - direction1 - direction2)) / (2 * delta))) / (2 * epsilon)
            return np.array(hessian_, dtype=np.float64)


class Base():

    def __init__(self, stop_condition: float = 1e-6, step: Union[float, str] = 'backtracking',
                 iteration_limit: int = int(1e6)):
        """
        Initialize optimizer
        :param stop_condition: gradient value at which the search is interrupted
        :param step: type of step length applied: float if fixed, 'backtracking', 'wolfe' or 'exact'
        :param iteration_limit: maximum iteration number before force quit
        """
        assert isinstance(stop_condition,
                          Union[float, int]) and stop_condition > 0, 'stop_condition must be a (samall) positive float'
        if not isinstance(step, float) and step not in ['backtracking', 'wolfe', 'exact']:
            raise ValueError("Step must be a float or ['backtracking', 'wolfe', 'exact']")
        assert isinstance(iteration_limit, int), 'iteration_limit must be int'

        self.logs = None
        self.stop = stop_condition
        self.step = step
        self.alpha, self.c1, self.c2, self.c = 1, 0.1, 0.9, 0.01
        self.auto_init, self.mode, self.contraction = 1, 'weak', 0.5
        self.iter_limit = iteration_limit

        self.parameters = dict({'Method': ''})
        self.parameters.update({'Stop condition': f'Module of gradient = {self.stop}'})
        self.parameters.update({'Iter limit': f'{self.iter_limit}'})
        self.task = None
        self.func = None
        return

    def reset(self):
        """
        Helper function that rewrites current parameters.
        """

        method, stop, limit = self.parameters['Method'], self.parameters['Stop condition'], self.parameters[
            'Iter limit']
        stop = stop.split('= ')[-1]
        self.parameters = dict({'Method': method})
        self.parameters.update({'Stop condition': f'Module of gradient = {stop}'})
        self.parameters.update({'Iter limit': f'{limit}'})

        if isinstance(self.step, float):
            self.parameters.update({'Step length search': f'fixed {self.step}'})
        elif self.step == 'backtracking':
            self.parameters.update({'Step length search': f'Backtracking'})
            self.parameters.update({'   Init. length': f'{self.alpha}'})
            self.parameters.update({'   Constant': f'{self.c}'})
            self.parameters.update({'   Contraction': f'{self.contraction}'})
        elif self.step == 'wolfe':
            self.parameters.update({'Step length search': f'Wolfe'})
            self.parameters.update({'   Mode': f'{self.mode}'})
            self.parameters.update({'   Initial length': f'{self.alpha}'})
            self.parameters.update({'   Constant c1': f'{self.c1}'})
            self.parameters.update({'   Constant c2': f'{self.c2}'})
        elif self.step == 'exact':
            self.parameters.update({'Step length search': f'exact'})
        elif self.step == 'Trust Region':
            self.parameters.update({'Step length search': 'Trust Region'})
            self.parameters.update({'   Radius': f'{self.trust_region}'})
            self.parameters.update({'   Eta': f'{self.eta}'})
            self.parameters.update({'   r': f'{self.r}'})

    def _finish(self, success: bool, func: Task, x: np.ndarray, der: np.ndarray, iteration: int, time_start: float):
        """
        Helper function to exit the iterative optimization algorithm.
        """

        print('..finished', end='\r')
        time_stop = time.perf_counter()

        if not success:

            self.parameters.update({'Status': f'Not Solved: {[f"{x_:.4g}" for x_ in x]}'})

        else:

            self.parameters.update({'Status': f'Solved: {[f"{x_:.4g}" for x_ in x]}'})

        self.parameters.update({'Runtime': f'{time_stop - time_start:.4g}s'})
        self.parameters.update({'Iterations': f'{iteration}'})

        if func.min is not None:
            distance = [(np.sum((x - minimizer) ** 2) ** 0.5, minimizer) for minimizer in func.min]
            distance.sort(key=lambda y: y[0])
            distance_min = distance[0]
            self.parameters.update(
                {'Dist. from closest min': f'{distance_min[0]:.4g} from {[f"{d:.4g}" for d in distance_min[1]]}'})
        self.parameters.update({'Last gradient module': f'{np.linalg.norm(der):.4g}'})

    def _get_step_length(self, func: Task, der: np.ndarray, x: np.ndarray, p: np.ndarray, q_matrix: np.ndarray = None):
        """
        Helper function, that searches for thr optimal step length. (Reiterates the algorithm according to parameters
        and hyperparameters)
        """

        if isinstance(self.step, float):
            step, step_iters = self.step, 0
        elif self.step == 'backtracking':
            step, step_iters = self.backtracking(func, x, der, p, self.alpha, self.c,
                                                 self.contraction, self.auto_init, self.logs)
        elif self.step == 'exact':
            step = 1.1 * np.sum(der * der) / np.sum(der * q_matrix @ der)
            step_iters = 0
        else:
            step, step_iters = self.wolfe(func, x, der, p, self.alpha, self.c1, self.c2, mode=self.mode,
                                          auto=self.auto_init, logs=self.logs)

        return step, step_iters

    def set_parameters(self, alpha_0: Union[int, float] = 1, c: float = 0.1, c1: float = 0.1,
                       c2: float = 0.9, mode: str = 'weak', contraction: float = 0.5, auto: int = 1):
        """
        Adjust the parameters before optimization
        :param alpha_0: initial step length: int, float or 'auto'
        :param c: backtracking constant (if applicable)
        :param c1: first Wolfe constant (if applicable)
        :param c2: second Wolfe constant (if applicable)
        :param mode: 'weak' or 'strong' Wolfe search (if applicable)
        :param contraction: backtracking contraction coefficient (if applicable)
        :param auto: initial step for auto-step
        :return:
        """

        assert isinstance(alpha_0, Union[int, float]) or alpha_0 == 'auto', \
            f'alpha_0 must be integer or float, not {type(alpha_0)}'
        assert isinstance(c, float), f'alpha_0 must be integer or float, not {type(c)}'
        assert 0 < c < 1, f'c must be in (0, 1)'
        assert isinstance(c1, float), f'alpha_0 must be integer or float, not {type(c1)}'
        assert 0 < c1 < 1, f'c must be in (0, 1)'
        assert isinstance(c2, float), f'alpha_0 must be integer or float, not {type(c2)}'
        assert 0 < c2 < 1, f'c must be in (0, 1)'
        assert isinstance(contraction, float), f'alpha_0 must be integer or float, not {type(contraction)}'
        assert 0 < contraction < 1, f'c must be in (0, 1)'
        assert mode in ('weak', 'strong'), f'mode must be "weak" or "strong", not {mode}'

        self.alpha = alpha_0
        self.auto_init = auto
        self.c = c
        self.c1 = c1
        self.c2 = c2
        self.mode = mode
        self.contraction = contraction

    def __str__(self):

        if 'Status' in self.parameters.keys() and self.parameters['Status'][0] == 'N':
            color = Fore.RED
        else:
            color = Fore.GREEN
        heading = f'{self.task.__name__}' if isinstance(self.task, Callable) else f'{self.task}'
        return (' OPTIMIZATION '.center(40, '-') + '\n' + 'Function'.ljust(35) + heading + '\n' +
                '\n'.join([':'.join([item[0].ljust(35), color + item[1] + Fore.BLACK])
                           for item in self.parameters.items()]) + '\n' + '-' * 40)

    @staticmethod
    def norm(x: Sequence) -> float:
        """
        Returns the absolute value of any sequence x
        :param x: sequence
        :return:
        """
        x = np.array(x)
        return np.sum(x ** 2) ** 0.5

    @staticmethod
    def auto_step(func: Task, derivative: np.ndarray, x_k: np.ndarray, x_k_1: np.ndarray, p: np.ndarray) -> float:
        """
        Provides estimate for initial step length to spare resources
        :param func: function in question
        :param derivative: gradient of the function
        :param x_k: current step point
        :param x_k_1: previous step point
        :param p: unit vector of gradient
        :return:
        """
        a = 2 * (func.value(x_k) - func.value(x_k_1))
        b = np.sum(derivative * p)
        a = np.abs(a / b)
        return a.astype(np.float64)

    @staticmethod
    def backtracking(func: Task, x: np.ndarray, gradient: np.ndarray, p: np.ndarray,
                     alpha_0: float = 1, c: float = 0.9, contraction: float = 0.5,
                     auto_init: int = 1, logs: pd.DataFrame = None) -> tuple:
        """
        Returns the quasi-optimal step length based on backtracking algorithm
        :param func: function in question
        :param x: current point
        :param gradient: derivative of the function
        :param p: unit vector of search direction
        :param alpha_0: step length to start with
        :param c: hyperparameter of backtracking
        :param contraction: rate of step length reduction
        :param auto_init: initial step for auto-step
        :param logs: dataframe of iterations to use
        :return: roughly optimizes the step length
        """
        coefficient = c * np.sum(p * gradient)  # slope of sufficient condition line
        if alpha_0 == 'auto' and len(logs) != 0:
            x_k_1 = np.array([float(x) for x in logs.iloc[len(logs) - 1]['x_k']])
            alpha = SteepestDecent.auto_step(func, gradient, x, x_k_1, p)
        elif alpha_0 == 'auto' and len(logs) == 0:
            alpha = auto_init
            print('no logs found')
        else:
            alpha = alpha_0
        iteration = 0

        while True:
            iteration += 1
            left_part = func.value(x + alpha * p)
            right_part = func.value(x) + alpha * coefficient
            if left_part <= right_part:
                break
            else:
                alpha *= contraction
            if iteration > 100:
                break
        return alpha, iteration

    @staticmethod
    def wolfe(func: Task, x: np.ndarray, gradient: np.ndarray, p: np.ndarray,
              alpha_0: float = 1, c1: float = 0.1, c2: float = 0.9, mode: str = 'strong',
              auto: int = 1, logs: pd.DataFrame = None) -> tuple:
        """

        :param func: function in question
        :param x: current pont
        :param gradient: gradient in the current point
        :param p: vector of direction
        :param alpha_0: initial step length
        :param c1: first Wolfe constant
        :param c2: second Wolfe constant
        :param mode: type of conditions: 'weak' (Wolfe) or 'strong' (Wolfe)
        :param auto: initial auto-step if applicable
        :param logs: dataframe of iterations to use
        :return: roughly optimizes the step length
        """
        assert alpha_0 == 'auto' or alpha_0 > 0, 'Step length cannot be negative'
        assert 0 < c1 < c2 < 1, f'Wolfe constants must be 0 < c1 < c2 < 1, {0} < {c1} < {c2} < {1} was given'
        assert mode in ('weak', 'strong'), 'Mode must be "weak" or "strong"'
        assert isinstance(func, Task), 'func must be of class Task'
        assert isinstance(gradient, np.ndarray), 'gradient must be numpy array'
        assert isinstance(p, np.ndarray), 'direction (p) must be numpy array'
        dims = x.shape[0]
        assert gradient.shape[0] == dims and p.shape[0] == dims, 'gradient or direction (p) of wrong shape'

        # Initial conditions for bisection search
        coefficient_1 = - c1 * np.sum(p * gradient)  # small negative number
        coefficient_2 = - c2 * np.sum(p * gradient)  # not so small negative number

        if alpha_0 == 'auto' and len(logs) > 1:
            x_k_1 = np.array([float(x) for x in logs.iloc[len(logs) - 1]['x_k']])
            alpha = SteepestDecent.auto_step(func, gradient, x, x_k_1, p)
        elif alpha_0 == 'auto':
            alpha = auto
            print('no logs')
        else:
            alpha = alpha_0

        # Starting points for bisection
        left_side = 0
        right_side = 100
        iteration = 0

        # First condition
        condition_1 = lambda a: func.value(x + a * p) - func.value(x) + a * coefficient_1  # <= 0 to hold
        while condition_1(alpha) > 0:
            iteration += 1
            right_side = alpha
            alpha = (right_side + left_side) / 2
            if iteration > 20:
                if alpha < 10 ** (-3) * np.linalg.norm(gradient):
                    return np.max([alpha, 10 ** (-2) * np.sum(gradient ** 2) ** 0.5]), iteration

        if mode == 'weak':
            return alpha, iteration

        # Try to assure the second condition
        for _ in range(20):
            if coefficient_2 < np.sum(p * func.grad(x + alpha * p)) < 0:
                break
            iteration += 1
            if np.sum(p * func.grad(x + alpha * p)) > 0 and mode == 'strong':
                right_side = alpha
                alpha = (right_side + left_side) / 2
                continue
            if np.sum(p * func.grad(x + alpha * p)) < coefficient_2:
                left_side = alpha
                alpha = (right_side + left_side) / 2
                continue
            break

        return alpha, iteration

    def visualize2d(self, minimizer: Sequence = None, save: bool = False):

        if len(self.logs) < 1:
            print('No iteration data found')
            return

        assert self.func is not None, 'There is no Task to visualize'
        func = self.func
        assert func.n_var == 2, f'Only 2D data can be visualized, not {func.n_var}D'
        if minimizer is None:
            minimizer = func.x0

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        trajectory = np.stack([func.x0, *[[np.float64(x) for x in pair] for pair in self.logs['x_k']]])
        heights = np.array([func.value(func.x0)] + [np.float64(x) for x in self.logs['f(x_k)']])
        ax.plot(trajectory[:, 0], trajectory[:, 1], heights, '-', color='black', alpha=1, markevery=1,
                markersize=3, marker='o')

        min_coord_x = min(min(func.x0), min(minimizer), np.min(trajectory[:, 0]))
        min_coord_y = min(min(func.x0), min(minimizer), np.min(trajectory[:, 1]))
        max_coord_x = max(max(func.x0), max(minimizer), np.max(trajectory[:, 0]))
        max_coord_y = max(max(func.x0), max(minimizer), np.max(trajectory[:, 1]))
        delta_x = (max_coord_x - min_coord_x) * 0.1
        delta_y = (max_coord_y - min_coord_y) * 0.1
        max_coord_x += delta_x
        max_coord_y += delta_y
        min_coord_x -= delta_x
        min_coord_y -= delta_y
        x_dim = np.linspace(min_coord_x, max_coord_x, 200)
        y_dim = np.linspace(min_coord_y, max_coord_y, 200)
        x_dim, y_dim = np.meshgrid(x_dim, y_dim)
        heights_f = np.array([[func.value([x_dim[i, j], y_dim[i, j]]) for j in range(200)] for i in range(200)])

        ax.plot_surface(x_dim, y_dim, heights_f, cmap='coolwarm', linewidth=0, antialiased=False, alpha=0.7)
        if save:
            plt.savefig(f'optim{datetime.datetime.now()}.png')
        plt.show()
        return

    def visualize1d(self, minimizer: Sequence = None, save: bool = False):

        if len(self.logs) < 1:
            print('No iteration data found')
            return

        assert self.func is not None, 'There is no Task to visualize'
        func = self.func
        assert func.n_var == 1, f'Only 1D data can be visualized, not {func.n_var}D'
        if minimizer is None:
            minimizer = func.x0

        fig, ax = plt.subplots()
        trajectory = np.stack([func.x0, *[np.float32(x) for x in self.logs['x_k']]])
        heights = np.array([func.value(func.x0)] + [np.float64(x) for x in self.logs['f(x_k)']])
        ax.plot(trajectory, heights, '-', color='black', alpha=1, markevery=1,
                markersize=3, marker='o')

        min_coord_x = min(min(func.x0), min(minimizer), np.min(trajectory)) - 0.3
        max_coord_x = max(max(func.x0), max(minimizer), np.max(trajectory)) + 0.3
        x_dim = np.linspace(min_coord_x, max_coord_x, 200)
        heights_f = np.array([func.value(x) for x in x_dim], dtype=np.float32)
        ax.plot(x_dim, heights_f)
        ax.set_title('Optimization steps')
        if save:
            plt.savefig(f'optim{datetime.datetime.now()}.png')
        plt.show()
        return

    @staticmethod
    def modify_hessian(hessian: np.ndarray):
        beta = 0.001
        new_hess = hessian
        if np.linalg.det(new_hess) > 0:
            return new_hess
        else:
            tau = -np.min(np.diag(new_hess)) + beta
            while True:
                new_hess = new_hess + tau * np.identity(len(new_hess))
                try:
                    np.linalg.cholesky(new_hess + tau * np.identity(len(new_hess)))
                    return new_hess
                except np.linalg.LinAlgError:
                    tau = np.max([2 * tau, beta])

    @staticmethod
    def pseudo_hessian(B: np.ndarray, H: np.ndarray, s: np.ndarray, y: np.ndarray, algorithm: str) -> tuple:
        if algorithm == 'SR1':
            try:
                B = B + ((y - B @ s) @ (y - B @ s).T) / ((y - B @ s).T @ s)
                H = H + ((s - H @ y) @ (s - H @ y).T) / ((s - H @ y).T @ y)
            except ZeroDivisionError:
                B, H = B, H
            return B, H
        if algorithm == 'DFP':
            rho = 1 / (y.T @ s)
            I = np.identity(len(B))
            try:
                B = (I - rho * y @ s.T) @ B @ (I - rho * s @ y.T) + rho * y @ y.T
                H = H - (H @ y @ y.T @ H) / (y.T @ H @ y) + (s @ s.T) / (y.T @ s)
            except ZeroDivisionError:
                B, H = B, H
            return B, H
        if algorithm == 'BFGS':
            rho = 1 / (y.T @ s)
            I = np.identity(len(B))
            try:
                B = B - (B @ s @ s.T @ B) / (s.T @ B @ s) + (y @ y.T) / (y.T @ s)
                H = (I - rho * s @ y.T) @ H @ (I - rho * y @ s.T) + rho * s @ s.T
            except ZeroDivisionError:
                B, H = B, H
            return B, H

    @staticmethod
    def beta(grad: np.ndarray, previous_grad: np.ndarray, previous_p: np.ndarray, algorithm: str = 'FR') -> float:
        if algorithm == 'FR':
            return np.sum(grad * grad) / np.sum(previous_grad * previous_grad)
        if algorithm == 'PR':
            return np.sum(grad * (grad - previous_grad)) / np.sum(previous_grad * previous_grad)
        if algorithm == 'HS':
            return np.sum(grad * (grad - previous_grad)) / - np.sum(previous_p * (grad - previous_grad))
        if algorithm == 'DY':
            return np.sum(grad * grad) / - np.sum(previous_p * (grad - previous_grad))


class SteepestDecent(Base):

    def __init__(self, stop_condition: float = 1e-6, step: Union[float, str] = 'backtracking',
                 iteration_limit: int = int(1e6)):
        """
        Initialize optimizer
        :param stop_condition: gradient value at which the search is interrupted
        :param step: type of step length applied: float if fixed, 'backtracking', 'wolfe' or 'exact'
        :param iteration_limit: maximum iteration number before force quit
        """

        super().__init__(stop_condition, step, iteration_limit)
        self.parameters['Method'] = 'Steepest Decent'

        return

    def optimize(self, func: Task, matrix: Sequence = None) -> tuple:
        """
        self.parameters attribute contains dict of all metadata of optimization process
        :param func: Task instance to optimize
        :param matrix: coefficients matrix if step is 'exact'
        :return: tuple(solution, logs of each iteration)
        """
        assert isinstance(func, Task), 'only Task instances can be accepted by optimizer'

        # Initial statements
        self.reset()
        self.func = func
        self.task = func.expression
        self.logs = pd.DataFrame(columns=['iteration', 'x_k', 'f(x_k)', 'gradient(x_k)',
                                          'direction_k', 'step_length', 'step_iterations'])
        x = np.array(func.x0, dtype=float)
        if func.min is not None:
            mins = np.array(func.min, dtype=float)
            assert x.shape == mins[0].shape, f'x_0 is {x.shape[0]}D while minimizers - {mins[0].shape[0]}D'
        self.parameters.update({'Starting point': f'{func.x0}'})
        time_start = time.perf_counter()
        iteration = 0
        p = np.zeros(len(x))

        # For exact step case
        q_matrix = None
        if self.step == 'exact':
            q_matrix = np.array(matrix, dtype=np.float32)
            assert q_matrix.shape == (func.n_var, func.n_var), (f'matrix has shape {q_matrix.shape}, '
                                                                f'{(func.n_var, func.n_var)} expected')

        # Iterations
        while True:
            iteration += 1

            # Step
            der = np.array(func.grad(x))  # vector gradient
            if np.linalg.norm(der) == 0:
                p = p * 0
            else:
                p = - der / np.linalg.norm(der)  # unit vector of gradient
            value = func.value(x)

            # Reading step value
            step, step_iters = self._get_step_length(func, der, x, p, q_matrix)

            # Interruption conditions (success or iteration limit)
            if self.stop > np.linalg.norm(der):
                self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                       [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                       'NO', 'NO']
                self._finish(True, func, x, der, iteration, time_start)
                break
            if iteration > self.iter_limit:
                self._finish(False, func, x, der, iteration, time_start)
                break

            # Update
            if iteration % 100 == 0:
                print(f'{iteration} it: gradient = {np.linalg.norm(der):.3g} ...', end='\r')
            self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                   [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                   f'{step:.4g}', step_iters]
            x = x + step * p

        return x, self.logs


class Newton(Base):

    def __init__(self, stop_condition: float = 1e-6, step: Union[float, str] = 'wolfe',
                 iteration_limit: int = int(1e5)):
        """
        Initialize optimizer
        :param stop_condition: gradient value at which the search is interrupted
        :param step: type of step length applied: float if fixed, 'backtracking', 'wolfe' or 'exact'
        :param iteration_limit: maximum iteration number before force quit
        """
        super().__init__(stop_condition, step, iteration_limit)

        assert isinstance(self.step, Union[float, int]) or self.step in ['backtracking',
                                                                         'wolfe'], 'exact step is not possible'

        self.parameters['Method'] = 'Newton'
        self.mode = 'strong'
        return

    def optimize(self, func: Task, modify_hessian: bool = True) -> tuple:
        """
        self.parameters attribute contains dict of all metadata of optimization process
        :param func: Task instance to optimize
        :param modify_hessian: enables hessian modification
        :return: tuple(solution, logs of each iteration)
        """
        assert isinstance(func, Task), 'only Task instances can be accepted by optimizer'

        # Initial statements
        self.reset()
        self.func = func
        self.task = func.expression
        self.logs = pd.DataFrame(columns=['iteration', 'x_k', 'f(x_k)', 'gradient(x_k)',
                                          'direction_k', 'step_length', 'step_iterations'])
        x = np.array(func.x0, dtype=float)
        if func.min is not None:
            mins = np.array(func.min, dtype=float)
            assert x.shape == mins[0].shape, f'x_0 is {x.shape[0]}D while minimizers - {mins[0].shape[0]}D'
        self.parameters.update({'Starting point': f'{func.x0}'})
        time_start = time.perf_counter()
        iteration = 0
        p = np.zeros(len(x))

        # Updating parameters
        self.parameters.update({'Modify Hessian': f'{modify_hessian}'})

        # Iterations
        while True:
            iteration += 1

            # Step
            der = np.array(func.grad(x))
            hess = self.modify_hessian(np.array(func.hess(x))) if modify_hessian else np.array(func.hess(x))
            try:
                p = np.linalg.solve(hess, -der)
                if np.linalg.norm(p) > 0:
                    p /= np.linalg.norm(p)
            except np.linalg.LinAlgError:
                hess[0, 0] = hess[0, 0] + np.abs(np.min(hess))
                p = np.linalg.solve(hess, -der)
                if np.linalg.norm(p) > 0:
                    p /= np.linalg.norm(p)

            # Step length
            step, step_iters = self._get_step_length(func, der, x, p)

            # Update
            x = x + step * p
            value = func.value(x)

            # Interruption conditions (success or iteration limit)
            if self.stop > np.sqrt(np.sum(der * der)):
                self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                       [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                       'NO', 'NO']
                self._finish(True, func, x, der, iteration, time_start)
                break
            if iteration > self.iter_limit:
                self._finish(False, func, x, der, iteration, time_start)
                break

            if iteration % 100 == 0:
                print(f'{iteration} it: gradient = {np.linalg.norm(der):.3g} ...', end='\r')
            self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                   [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                   f'{step:.4g}', step_iters]

        return x, self.logs


class ConjugateGradient(Base):

    def __init__(self, stop_condition: float = 1e-6, step: Union[float, str] = 'wolfe',
                 iteration_limit: int = int(1e4)):
        """
        Initialize optimizer
        :param stop_condition: gradient value at which the search is interrupted
        :param step: type of step length applied: float if fixed, 'backtracking', 'wolfe' or 'exact'
        :param iteration_limit: maximum iteration number before force quit
        """
        super().__init__(stop_condition, step, iteration_limit)

        self.parameters['Method'] = 'Conjugate Gradient'
        self.mode = 'strong'
        return

    def linear(self, func: Task, a_matrix: Sequence[Sequence] = None, b_matrix: Sequence = None) -> tuple:
        """
        self.parameters attribute contains dict of all metadata of optimization process
        :param func: Task instance to optimize
        :param a_matrix: matrix of coefficients, only required for 'linear' algorithm
        :param b_matrix: matrix of free coefficients, only required for 'linear' algorithm
        :return: tuple(solution, logs of each iteration)
        """
        assert isinstance(func, Task), 'only Task instances can be accepted by optimizer'
        assert a_matrix is not None, 'A matrix must be provided for linear CG'
        assert b_matrix is not None, 'b matrix must be provided for linear CG'
        assert isinstance(a_matrix, Sequence | np.ndarray) and (len(a_matrix) == func.n_var), \
            f'"a_matrix" has length ({len(a_matrix)}), ({func.n_var}) was expected'
        assert isinstance(b_matrix, Sequence | np.ndarray) and len(b_matrix) == func.n_var, \
            f'"b_matrix" has length ({len(b_matrix)}), ({func.n_var}) was expected'

        # Initial statements
        self.task = f'Linear task with {func.n_var} parameters'
        self.func = func
        self.logs = pd.DataFrame(columns=['iteration', 'x_k', 'f(x_k)', 'gradient(x_k)/residual',
                                          'direction_k', 'beta', 'step_length', 'step_iterations'])
        x = func.x0
        if func.min is not None:
            mins = np.array(func.min, dtype=float)
            assert x.shape == mins[0].shape, f'x_0 is {x.shape[0]}D while minimizers - {mins[0].shape[0]}D'
        self.parameters.update({'Starting point': f'{func.x0}'})
        self.parameters.update({'Algorithm': f'linear'})
        time_start = time.perf_counter()
        iteration = 0

        # Initial conditions for linear case
        a, b = np.array(a_matrix, dtype=np.float64), np.array(b_matrix, dtype=np.float64)
        residual = a @ x - b
        direction = -residual

        # Iterations
        while True:
            iteration += 1

            # Step
            step = np.sum(residual.T @ residual) / (direction.T @ a @ direction)
            x = x + step * direction
            residual_ = residual + step * a @ direction
            beta = np.sum(residual_ * residual_) / np.sum(residual * residual)
            direction = -residual_ + beta * direction
            residual = residual_
            value = func.value(x)

            # Interruption conditions
            if self.stop > np.linalg.norm(residual):
                self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                       [f'{r_:.4g}' for r_ in residual], [f'{d_:.4g}'
                                                                                          for d_ in direction],
                                                       '-', 'NO', 'NO']
                self._finish(True, func, x, residual, iteration, time_start)
                break
            if iteration == self.iter_limit:
                self._finish(False, func, x, residual, iteration, time_start)
                break

            if iteration % 100 == 0:
                print(f'{iteration} it: gradient = {np.linalg.norm(residual):.3g} ...', end='\r')
            self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                   [f'{r_:.4g}' for r_ in residual], [f'{d_:.4g}' for d_ in direction],
                                                   f'{beta:.4g}', f'{step:.4g}', '1']

        return x, self.logs

    def optimize(self, func: Task, algorithm: str = 'FR',
                 a_matrix: Sequence[Sequence] = None, b_matrix: Sequence = None) -> tuple:
        """
        self.parameters attribute contains dict of all metadata of optimization process
        :param func: Task instance to optimize
        :param algorithm: type of updating algorithm: 'FR' for Fletcher-Reeves / 'PR' for Polak-Ribiere /
        'HS' for Hestenes-Stiefel / 'DY' for Dai-Yuan / 'linear' for linear CG
        :param a_matrix: matrix of coefficients, only required for 'linear' algorithm
        :param b_matrix: matrix of free coefficients, only required for 'linear' algorithm
        :return: tuple(solution, logs of each iteration)
        """

        assert isinstance(func, Task), 'only Task instances can be accepted by optimizer'
        assert algorithm in ('FR', 'PR', 'HS', 'DY', 'linear'), "Algorithms must be: 'FR', 'PR', 'HS', 'DY' or 'linear'"
        if algorithm == 'linear':
            assert self.step == 'exact'
            return self.linear(func, a_matrix, b_matrix)

        # Initial statements
        self.reset()
        self.task = func.expression
        self.func = func
        self.logs = pd.DataFrame(columns=['iteration', 'x_k', 'f(x_k)', 'gradient(x_k)',
                                          'direction_k', 'beta', 'step_length', 'step_iterations'])
        self.mode = 'strong'
        x = np.array(func.x0)
        if func.min is not None:
            mins = np.array(func.min, dtype=float)
            assert x.shape == mins[0].shape, f'x_0 is {x.shape[0]}D while minimizers - {mins[0].shape[0]}D'
        self.parameters.update({'Starting point': f'{func.x0}'})
        self.parameters.update({'Algorithm': f'{algorithm}'})
        time_start = time.perf_counter()
        iteration = 0

        # Initial conditions
        beta, der_previous, p = 0, None, None

        # Iterations
        while True:
            iteration += 1

            # Step
            der = np.array(func.grad(x))
            if iteration == 1:
                p = - der
                p /= np.linalg.norm(p)
            elif iteration > 1:
                beta = self.beta(der, der_previous, p, algorithm=algorithm)
                p = - der + beta * p
            der_previous = der

            # Step length
            step, step_iters = self._get_step_length(func, der, x, p)

            # Update
            x = x + step * p
            value = func.value(x)

            # Iterruption conditions (success or iteration limit)
            if self.stop > np.linalg.norm(der):
                self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                       [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                       '-', 'NO', 'NO']
                self._finish(True, func, x, der, iteration, time_start)
                break
            if iteration == self.iter_limit:
                self._finish(False, func, x, der, iteration, time_start)
                break

            if iteration % 100 == 0:
                print(f'{iteration} it: gradient = {np.linalg.norm(der):.3g} ...', end='\r')
            self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                   [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                   f'{beta:.4g}', f'{step:.4g}', step_iters]

        return x, self.logs


class QuasiNewton(Base):

    def __init__(self, stop_condition: float = 1e-6, step: Union[float, str] = 'wolfe',
                 iteration_limit: int = int(1e4)):
        """
        Initialize optimizer
        :param stop_condition: gradient value at which the search is interrupted
        :param step: type of step length applied: float if fixed, 'backtracking', 'wolfe' or 'exact'
        :param iteration_limit: maximum iteration number before force quit
        """
        super().__init__(stop_condition, step, iteration_limit)

        assert isinstance(self.step, Union[float, int]) or self.step in ['backtracking',
                                                                         'wolfe'], 'exact step is not possible'

        self.parameters['Method'] = 'Quasi Newton'
        self.mode = 'strong'
        self.eta, self.r, self.trust_region = 0.00001, 1e-7, 10
        return

    def optimize(self, func: Task, algorithm: str = 'SR1') -> tuple:
        """
        self.parameters attribute contains dict of all metadata of optimization process
        :param func: Task instance to optimize
        :param algorithm: type of updating algorithm: 'SR1' / 'DFP' / 'BFGS' / 'SR1_trust_region'
        :return: tuple(solution, logs of each iteration)
        """
        assert isinstance(func, Task), 'only Task instances can be accepted by optimizer'
        assert algorithm in ('SR1', 'DFP', 'BFGS', 'SR1_trust_region'), ("algorithm must be 'SR1', 'DFP', "
                                                                         "'BFGS' or 'SR1_trust_region'")
        if algorithm == 'SR1_trust_region':
            self.step = 'Trust Region'

        # Initial statements
        self.reset()
        self.func = func
        self.task = func.expression
        self.logs = pd.DataFrame(columns=['iteration', 'x_k', 'f(x_k)', 'gradient(x_k)',
                                          'direction_k', 'B', 'H', 'step_length', 'step_iterations'])
        self.mode = 'strong'
        x = np.array(func.x0)
        if func.min is not None:
            mins = np.array(func.min, dtype=float)
            assert x.shape == mins[0].shape, f'x_0 is {x.shape[0]}D while minimizers - {mins[0].shape[0]}D'
        self.parameters.update({'Starting point': f'{func.x0}'})
        self.parameters.update({'Algorithm': f'{algorithm}'})
        time_start = time.perf_counter()
        iteration = 0

        # Initial conditions
        if algorithm == 'SR1_trust_region':
            return self.__sr1_trust_region(func)
        pseudo_hess, pseudo_hess_inverted = np.identity(func.n_var), np.identity(func.n_var)
        self.eta, self.r, self.trust_region = 0.00001, 1e-7, 10

        # Iterations
        while True:
            iteration += 1

            # Step
            der = np.array(func.grad(x))
            p = - pseudo_hess_inverted @ der
            if algorithm == 'SR1':
                step, step_iters = 1, 1
            else:
                step, step_iters = step, step_iters = self._get_step_length(func, der, x, p)
            x_next = x + step * p
            s = (x_next - x).reshape(-1, 1)
            y = (np.array(func.grad(x_next)) - der).reshape(-1, 1)
            pseudo_hess, pseudo_hess_inverted = self.pseudo_hessian(pseudo_hess, pseudo_hess_inverted, s, y,
                                                                    algorithm=algorithm)
            x = x_next
            value = func.value(x)

            # Interruption conditions (success or iteration limit)
            if self.stop > np.linalg.norm(der):
                self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                       [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                       '-',
                                                       '-',
                                                       'NO', 'NO']
                self._finish(True, func, x, der, iteration, time_start)
                break
            if iteration == self.iter_limit:
                self._finish(False, func, x, der, iteration, time_start)
                break

            if iteration % 100 == 0:
                print(f'{iteration} it: gradient = {np.linalg.norm(der):.3g} ...', end='\r')
            self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                   [f'{der_:.4g}' for der_ in der], [f'{p_:.4g}' for p_ in p],
                                                   [[f'{x_:.4g}' for x_ in x] for x in pseudo_hess],
                                                   [[f'{x_:.4g}' for x_ in x] for x in pseudo_hess_inverted],
                                                   f'{step:.4g}', step_iters]

        return x, self.logs

    def __sr1_trust_region(self, func: Task) -> tuple:

        self.task = func.expression
        self.func = func
        self.logs = pd.DataFrame(columns=['iteration', 'x_k', 'f(x_k)', 'gradient(x_k)',
                                          'B', 'step', 'step_length', 'trust region', 'ratio'])
        iteration = 0
        x = func.x0

        if func.min is not None:
            mins = np.array(func.min, dtype=float)
            assert x.shape == mins[0].shape, f'x_0 is {x.shape[0]}D while minimizers - {mins[0].shape[0]}D'

        self.parameters.update({'Algorithm': 'SR1'})
        self.parameters.update({'Starting point': f'{func.x0}'})
        time_start = time.perf_counter()
        pseudo_hess, pseudo_hess_inverted = np.identity(func.n_var), np.identity(func.n_var)

        s = np.zeros(len(x))
        while True:
            iteration += 1

            # Step
            der = func.grad(x)
            s = np.linalg.solve(pseudo_hess, -der)
            if np.linalg.norm(s) > self.trust_region:
                s = 0.8 * s * self.trust_region / np.linalg.norm(s)
            y = (np.array(func.grad(x + s) - der)).reshape(-1, 1)
            actual_red = (func.value(x) - func.value(x + s))
            predicted_red = - (np.sum(der * s) + 0.5 * np.sum(s * pseudo_hess @ s))
            ratio = actual_red / predicted_red if predicted_red != 0 else 1

            if ratio < 0.2:
                self.trust_region = np.max([0.99 * self.trust_region, 10 ** (-7)])
            if ratio > 1:
                self.trust_region *= 1.001
            if ratio > self.eta:
                x += s
                step = s
            else:
                x = x
                step = np.zeros(func.n_var)

            if np.abs((y - pseudo_hess @ s.reshape(-1, 1)).T @ s) > self.r:
                pseudo_hess, pseudo_hess_inverted = self.pseudo_hessian(pseudo_hess, pseudo_hess_inverted,
                                                                        - s.reshape(-1, 1), - y, algorithm='SR1')
            value = func.value(x)

            # Interruption conditions (success or iteration limit)
            if self.stop > np.linalg.norm(der):
                self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                       [f'{der_:.4g}' for der_ in der], '-', '-', 'NO', 'NO',
                                                       f'{ratio:.4g}']
                self._finish(True, func, x, der, iteration, time_start)
                break
            elif iteration == self.iter_limit:
                self._finish(False, func, x, der, iteration, time_start)
                break


            elif np.linalg.norm(step) != 0:
                if iteration % 100 == 0:
                    print(f'{iteration} it: gradient = {np.linalg.norm(der):.3g} ...', end='\r')
                self.logs.loc[len(self.logs.index)] = [iteration, [f'{x_:.4g}' for x_ in x], f'{value:.4g}',
                                                       [f'{der_:.4g}' for der_ in der],
                                                       [[f'{x_:.4g}' for x_ in x] for x in pseudo_hess],
                                                       [f'{s_:.4g}' for s_ in step],
                                                       f'{np.linalg.norm(step):.4g}', f'{self.trust_region:.4g}',
                                                       f'{ratio:.4g}']

        return x, self.logs


if __name__ == '__main__':
    task = Task('100 * (x_1 - x_2^2)^2 + (1 - x_2)^2', x0=[2, 0])
    print(task)

    Newton_opt = Newton()
    QN_opt = QuasiNewton()
    CG_opt = ConjugateGradient()

    Newton_opt.optimize(task, modify_hessian=True)
    print(Newton_opt)

    QN_opt.optimize(task, algorithm='DFP')
    print(QN_opt)

    CG_opt.optimize(task, algorithm='PR')
    print(CG_opt)
    CG_opt.visualize2d()