import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from typing import Callable, Union, Sequence, Optional
from scipy.integrate import nquad
from scipy.optimize import minimize
from tqdm import tqdm
import inspect
from scipy.stats import norm, describe
from warnings import simplefilter
simplefilter('ignore')

def mci(function: Callable,
        pdf: Callable,
        domain: Union[list[tuple], tuple],
        validate_pdf: bool = True,
        sampling_function: Optional[Callable] = None,
        n_samples: int = 100,
        plot: bool = False,
        animation: bool = False,
        verbose: bool = True,
        seed: int = 1234) -> tuple[float, float]:
    """
    Monte Carlo Integration

    1. Assertion statements for the input
    2. Validation of the PDF function (density function) and its adjustment if needed
    3. Drawing samples from the adjusted PDF using Rejection Sampling
    4. Evaluating the MC estimator

    Args:
        function (Callable): function to integrate (integrand)
        pdf (Callable): distributions to draw sample points from
        domain (Union[list[tuple], tuple]): list of integration boundaries for each variable
        validate_pdf (bool): toogle to check the validity of a provided distribution. Defaults to False.
        sampling_function(Callable, optional): a function which provides draws directly (element-wise or as a list - for multivariate)
        n_samples (int, optional): number of samples to use. Defaults to 100.
        plot (bool, optional): toggle to plot the result. Defaults to False.
        animation (bool, optional): toggle to animate the result. Defaults to False.
        verbose (bool, optional): activate the verbose mode - print down more details. Defaults to True.
        seed (int, optional): numpy random seed for reproduceability. Defaults to 1234.

    Raises:
        ValueError: if PDF is not appropriate for the task
        AssertionError: if the input is not valid

    Returns:
        tuple[float, float]: MC estimator, Reference value (obtained by scipy.nquad)
    """

    #####################################
    # Assertion statements
    #####################################

    assert isinstance(function, Callable), 'function must be callable'
    assert isinstance(pdf, Callable), 'density function must be callable'
    assert isinstance(domain, tuple) or isinstance(domain, list), 'function must be callable'
    assert isinstance(n_samples, int) and n_samples > 0, 'n_vars must be a positive integer'
    assert isinstance(verbose, bool), 'verbose must be boolean'
    assert isinstance(validate_pdf, bool), 'validate_pdf must be boolean'
    assert sampling_function is None or isinstance(sampling_function, Callable), 'sampling function must be callable'
    f_args = np.sum([str(a.default) == "<class 'inspect._empty'>"
                     for _, a in inspect.signature(function).parameters.items()])
    p_args = np.sum([str(a.default) == "<class 'inspect._empty'>"
                     for _, a in inspect.signature(pdf).parameters.items()])
    assert f_args == p_args, f'function takes {f_args} positional argument(s) but PDF takes {p_args}'
    n_vars = f_args

    # Asserting that each variable has lower and upper boundary
    if n_vars == 1:
        if not isinstance(domain[0], Sequence):
            assert len(domain) == 2, 'each variable must have lower and upper bounds'
            domain = [domain]
        else:
            assert len(domain) == 1, f'{n_vars} variable(s), but {len(domain)} domain(s) provided'
    else:
        assert len(domain) == n_vars, f'n_vars = {n_vars} but {len(domain)} boundaries provided'
        for bounds in domain:
            assert len(bounds) == 2, 'each variable must have lower and upper bounds'
    infinite_domain = False

    # Asserting that upper boundary is indeed larger than lower boundary
    for bounds in domain:
        assert bounds[0] < bounds[1], f'domain {bounds} is not valid'
        if bounds[0] == -np.inf or bounds[1] == np.inf:
            infinite_domain = True
    assert isinstance(seed, int) and seed > 0, 'seed must be a positive integer'
    np.random.seed(seed)

    # Format for tqdm progress bar
    bar_format = "\033[30m{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\033[0m"

    #####################################
    # Reporting the task
    #####################################
    if verbose:
        function_code = inspect.getsource(function).replace('\n', '').replace('return', '')
        function_code = ''.join(function_code.split(':')[1:])
        print('Task:')
        print('Integrand:'.ljust(30), re.sub(r' {1,100}', ' ', function_code))
        pdf_code = inspect.getsource(pdf).replace('\n', '').replace('return', '')
        pdf_code = ''.join(pdf_code.split(':')[1:])
        print('Raw PDF:'.ljust(30) + re.sub(r' {1,100}', ' ', pdf_code))
        print('PDF validation required:'.ljust(30) + str(validate_pdf))
        if sampling_function:
            try:
                sampler_code = inspect.getsource(sampling_function).replace('\n', '').replace('return', '')
                sampler_code = ''.join(sampler_code.split(':')[1:])
            except TypeError:
                sampler_code = 'Provided'
        else:
            sampler_code = 'Not Provided'
        print('Sampling function provided:'.ljust(30) + re.sub(r' {1,100}', ' ', sampler_code))
        print('Number of variables:'.ljust(30) + str(n_vars))
        print('Boundaries of integration:'.ljust(30) + ', '.join([f'[{b[0]}, {b[1]}]' for b in domain]))
        print('Infinite domain:'.ljust(30) + str(infinite_domain))
        print('Number of samples:'.ljust(30) + str(n_samples))
        print('Plotting option:'.ljust(30) + str(plot))
        print('Animation:'.ljust(30) + str(animation) + '\n')


    #####################################
    # Checking if PDF is valid
    #####################################
    if validate_pdf:
        if verbose:
            print('Checking PDF:')

        # Finding the infinum of PDF: if infinum is smaller than 0, PDF will be increased by the necessary value 
        # to sure that it is non-negative over the domain
        pdf_min_ = minimize(lambda x: pdf(*x),
                            x0=np.array([d[0] if d[0] != -np.inf else -20 for d in domain]),
                            method="trust-constr", bounds=domain)
        pdf_min = pdf(*pdf_min_.x)
        pdf_min_ = minimize(lambda x: pdf(*x),
                            x0=np.array([d[1] if d[0] != -np.inf else 20 for d in domain]),
                            method="trust-constr", bounds=domain)
        pdf_min = min(pdf_min, pdf(*pdf_min_.x))
        if verbose:
            print('PDF minimum:'.ljust(30) + str(pdf_min))
        if pdf_min < 0 and verbose:
            print('PDF reaches negative value:'.ljust(30) + f'{pdf_min:.3g}'), 
            print('  it will be adjusted by:'.ljust(30) + f'{-pdf_min:.3g}')

        # Additional 'volume' to add to the PDF integral
        pdf_min = np.min([0, pdf_min])
        volume = 1
        for bounds in domain:
            volume *= (bounds[1] - bounds[0])
        add_volume = - volume * pdf_min if pdf_min != 0 else 0

        # Asserting that PDF integrates to 1 over the domain. If not: it is multiplied by the required factor
        # PDF -> adjusted PDF
        pdf_integral = nquad(pdf, domain)[0] + add_volume
        if pdf_integral < 0:
            raise ValueError('PDF is divergent over the domain')
        if not np.isclose(pdf_integral, 1.0, rtol=1e-8):
            if verbose:
                print('PDF integrates to:'.ljust(30) + f'{pdf_integral:.3g}')
                print('  it will be multiplied by:'.ljust(30) + f'{1/pdf_integral:.3g}')
            pdf_adjusted = lambda x: (pdf(*x) - pdf_min) / pdf_integral
        else:
            pdf_adjusted = lambda x: pdf(*x) - pdf_min
    else:
        pdf_adjusted = lambda x: pdf(*x)

    #####################################
    # Rejection Sampling from PDF
    #####################################
    draws = []
    
    if not sampling_function:
        if verbose:
            print('\nRejection Sampling from PDF:')

        # Defining the propasal distribution q(x) for rejection sampling: uniform for finite domain / broad normal for infinite domain
        if infinite_domain:
            if verbose:
                print('Proposal (q(x)):'.ljust(30) + 'normal')
            pseudo_domain = [(bounds[0] if bounds[0] > -np.inf else 0, bounds[1] if bounds[1] < np.inf else 20) for bounds in domain]
            pseudo_min1 = minimize(lambda x: pdf_adjusted(x), x0=np.array([b[1] for b in pseudo_domain]), method="Nelder-Mead", bounds=pseudo_domain)
            pseudo_min2 = minimize(lambda x: pdf_adjusted(x), x0=np.array([b[0] for b in pseudo_domain]), method="Nelder-Mead", bounds=pseudo_domain)
            pseudo_max1 = minimize(lambda x: -pdf_adjusted(x), x0=np.array([b[1] for b in pseudo_domain]), method="Nelder-Mead", bounds=pseudo_domain)
            pseudo_max2 = minimize(lambda x: -pdf_adjusted(x), x0=np.array([b[0] for b in pseudo_domain]), method="Nelder-Mead", bounds=pseudo_domain)
            pseudo_min = np.min([pdf_adjusted(pseudo_min1.x.tolist()), pdf_adjusted(pseudo_min2.x.tolist())])
            pseudo_max = np.max([pdf_adjusted(pseudo_max1.x.tolist()), pdf_adjusted(pseudo_max2.x.tolist())])
            s = pseudo_max / (pseudo_min + 1e-10)
            s = np.max([np.log10(s), 1])
            q = lambda x: norm.pdf(x, 0, 100 / s)[0]
            q_sample = lambda: np.random.normal(0, 100 / s)
        else:
            if verbose:
                print('Proposal (q(x)):'.ljust(30) + 'uniform')
            q = lambda x: 1
            q_sample = lambda a, b: np.random.uniform(a, b)

        # Upper bound estimation for p(x) / q(x)
        M = minimize(lambda x: q(x) / pdf_adjusted(x), x0=np.zeros(n_vars).tolist(), method="Nelder-Mead", bounds=domain)
        M = M.x.tolist()
        M = pdf_adjusted(M) / q(M)
        if verbose:
            print('pdf(x) / q(x) upper bound:'.ljust(30) + f'{M:.2g}')

        # Drawing samples from p(x) using rejection sampling
        proposals = 0
        with tqdm(total=n_samples, desc='Collecting draws from PDF: ', bar_format=bar_format) as pbar:
            while len(draws) < n_samples:
                proposals += 1
                sample = []
                while len(sample) < n_vars:
                    new = q_sample() if infinite_domain else q_sample(domain[len(sample)][0], domain[len(sample)][1])
                    if domain[len(sample)][0] <= new <= domain[len(sample)][1]:
                        sample.append(new)
                criterium = np.random.uniform(0, 1)
                if criterium < pdf_adjusted(sample) / (M * q(sample)):
                    draws.append(sample)
                    pbar.update(1)
        a_rate = n_samples / proposals
        if verbose:
            print('Acceptance rate:'.ljust(30) + f'{a_rate:.3g}')
            draws_desc = describe(draws)
            print('Ranges:'.ljust(30) + str([f'({a:.3g}, {b:.3g})' for a, b in zip(draws_desc.minmax[0], draws_desc.minmax[1])]))
            print('Mean(s):'.ljust(30) + f'{draws_desc.mean}')
            print('Variance(s):'.ljust(30) + f'{draws_desc.variance}\n')
    else:
        q_sample = lambda: sampling_function()
        if not (isinstance(q_sample(), Sequence) or isinstance(q_sample(), np.ndarray)):
            element_wise_sampler = True
        else:
            assert len(q_sample()) == n_vars, 'length of sampling function output is not equal to the number of variables'
            element_wise_sampler = False
        with tqdm(total=n_samples, desc='Collecting draws from the sampling function: ', bar_format=bar_format) as pbar:
            while len(draws) < n_samples:
                if element_wise_sampler:
                    sample = []
                    while len(sample) < n_vars:
                        new = q_sample()
                        if domain[len(sample)][0] <= new <= domain[len(sample)][1]:
                            sample.append(new)
                    draws.append(sample)
                    pbar.update(1)
                else:
                    sample = q_sample()
                    if not all(domain[i][0] <= sample[i] <= domain[i][1] for i in range(n_vars)):
                        continue
                    draws.append(sample)
                    pbar.update(1)
        if verbose:
            draws_desc = describe(draws)
            print('Ranges:'.ljust(30) + str([f'({a:.3g}, {b:.3g})' for a, b in zip(draws_desc.minmax[0], draws_desc.minmax[1])]))
            print('Mean(s):'.ljust(30) + f'{draws_desc.mean}')
            print('Variance(s):'.ljust(30) + f'{draws_desc.variance}\n')

    #####################################
    # Monte Carlo Estimation
    #####################################
    if verbose:
        print('Monte Carlo estimation:')

    # Computing MC estimator
    mc_estimator = []
    f_values = []
    mc_sum = 0
    counter = 0
    with tqdm(total=len(draws), desc='Computing MC estimator: ', bar_format=bar_format) as pbar:
        for draw in draws:
            pbar.update(1)
            p_value = pdf_adjusted(draw)
            f_value = function(*draw)
            if p_value == 0:
                continue
            is_ratio = function(*draw) / p_value
            mc_sum += is_ratio
            counter += 1
            mc_estimator.append(mc_sum / counter)
            f_values.append(f_value)

    # Additional parameters for MC estimator
    variance = np.var(f_values) / len(f_values)
    sd = np.std(f_values) / np.sqrt(len(f_values))
    if verbose:
        print('Number of omitted samples:'.ljust(30) + f'{n_samples - counter}')
        print('Variance:'.ljust(30) + f'{variance:.5g}')
        print('SD:'.ljust(30) + f'{sd:.5g}')
    final_estimator = mc_estimator[-1]
    print('The final estimation:'.ljust(30) + f'{final_estimator:.3g}')
    print('Confidence interval:'.ljust(30) + f'({final_estimator - 1.96 * sd:.5g} ... {final_estimator + 1.96 * sd:.5g}) p=0.05')

    # Using quadrature (scipy.integrate.nquad) as a reference vlue for comparison
    ref = nquad(function, domain)[0]
    print('Reference value'.ljust(30) + f'{ref:.5g}')
    print('Error:'.ljust(30) + f'{final_estimator - ref:.5g}')


    #####################################
    # Plotting Results
    #####################################
    if plot:

        # For univariate case: plot f(x), PDF(x), adjusted PDF(x) and the MC estimator through iterations
        if n_vars == 1:
            
            # Limiting of visible domain if it is infinite:
            if domain[0][0] == -np.inf:
                if domain[0][1] == np.inf:
                    domain = [(-10, 10)]
                else:
                    domain = [(domain[0][1] - 20, domain[0][1])]
            else:
                if domain[0][1] == np.inf:
                    domain = [(domain[0][0], domain[0][0] + 20)]

            x = np.linspace(domain[0][0], domain[0][1], 100)
            y = np.vectorize(f)(x)
            p_ = np.vectorize(pdf)(x)
            p_a = np.vectorize(lambda x: pdf_adjusted([x]))(x)

            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, :])

            ax0.plot(x, y, color='blue')
            ax0.set_title('f(x)')
            delta = 0.3 * (np.max([0, np.max(y)]) - np.min([0, np.min(y)]))
            ax0.set_ylim((np.min([0, np.min(y)]) - delta, np.max([0, np.max(y)]) + delta))
            p = plt.Polygon([(domain[0][0], 0), *list(zip(x, y)), (domain[0][1], 0)], color='lightblue')
            ax0.add_patch(p)
            ax0.plot(x, [0] * len(x), color='grey', linestyle='--')

            ax1.set_title('PDF')
            ax1.plot(x, p_, color='red', label = 'raw')
            if np.any(p_ != p_a):
                ax1.plot(x, p_a, color='orange', label = 'adjusted')
                ax1.legend()
            delta = 0.3 * (np.max([0, np.max(p_), np.max(p_a)]) - np.min([0, np.min(p_), np.min(p_a)]))
            ax1.set_ylim((np.min([0, np.min(p_), np.min(p_a)]) - delta, np.max([0, np.max(p_), np.max(p_a)]) + delta))
            ax1.plot(x, [0] * len(x), color='grey', linestyle='--')

            ax2.set_title('MC Estimator')
            ax2.plot(list(range(1, counter + 1)), mc_estimator, color='black',
                     label=f'MC estimate\nfinal = {mc_estimator[-1]:.5g}')
            ax2.plot(list(range(1, counter + 1)), [ref]*counter, color='grey', label=f'Quad estimate = {ref:.5g}')
            ax2.legend()

            plt.show()
        else:

            # For multivariate case only MC estimator is plotted:

            fig, axs = plt.subplots(1, 1, figsize=(10, 4))
            axs.set_title('MC Estimator')
            axs.plot(list(range(1, counter + 1)), mc_estimator, color='black',
                     label=f'MC estimate \n final = {mc_estimator[-1]:.5g}')
            axs.plot(list(range(1, counter + 1)), [ref] * counter, color='grey', label=f'Quad estimate = {ref:.5g}')
            axs.legend()

            plt.show()

    #####################################
    # Animation
    #####################################
    if animation:

        window = n_samples // 10
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))
        axs.set_title('MC Estimator')
        plot = axs.plot([0], mc_estimator[0], color='black', label=f'MC estimate')[0]
        axs.plot(list(range(0, counter + 100)), [ref] * (counter + 100), color='grey', label=f'Quad estimate = {ref:.5g}')
        axs.set_ylim((min(mc_estimator) - 1, max(mc_estimator) + 1))
        axs.set_xlim((0, 2 * window))
        axs.legend()

        data = []
        for i in range(1, len(mc_estimator)):
            min_ = min(mc_estimator[max(0, i-window):min(i+window, len(mc_estimator))])
            max_ = max(mc_estimator[max(0, i-window):min(i+window, len(mc_estimator))])
            range_ = max_ - min_
            min_ = np.min([ref - range_ / 10, min_ - range_ / 10])
            max_ = np.max([ref + range_ / 10, max_ + range_ / 10])
            delta = np.max([np.abs(ref - min_), np.abs(ref - max_)])
            limits = ref - delta, ref + delta
            data.append((i - 1, mc_estimator[i], limits))

        def init():
            return []

        def update(data: tuple):
            
            iteration, current, limits = data

            plot.set_xdata(range(iteration))
            plot.set_ydata(mc_estimator[:iteration])
            plot.set(label = f'MC estimate = {current:.5g}')
            if iteration > 1:
                axs.set_xlim((max(0, iteration - window), max(iteration + window, 2 * window)))
            if iteration % window == 0:
                axs.set_ylim(limits)
            axs.legend()

            return plot,
    
        animation = FuncAnimation(fig, update, frames=data, interval=int(1000 * 45 / n_samples), blit=False, init_func=init)
        return HTML(animation.to_html5_video())
    
    return mc_estimator[-1], ref


if __name__ == '__main__':

    def f(x):
        return np.sin(x) + np.cos(x)

    def pdf_1(x):
        return - (x + 1) * (x - 2.5)

    def pdf_2(x):
        return 1

    def pdf_3(x):
        return 0.5 * x + 1 if x < 1 else 2.5 - x

    mci(f, pdf_1, [(0, 2)], n_samples=1000, plot=True, verbose=True)
    mci(f, pdf_2, [(0, 2)], n_samples=1000, plot=True, verbose=True)
    mci(f, pdf_3, [(0, 2)], n_samples=1000, plot=True, verbose=True)
