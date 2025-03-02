# math_tips

Some math functions which I found interesting to implement. Mostly about Calculus.

## double_integral

As the name suggests, this function is made to compute and visualize double integrals (with non-rectangular domains).
It does not pretend to efficiency or brevity. The main purpose was to create a simple self-explainable code
and a clear visualization.

Example:
```python
double_integral(lambda x, y: 1 / (3 * x + 2 * y), ['x < y < 5 * x', '2 < x + 2 * y < 4'],
                plot_on=True, precision=300)
```
```
   Precision = 300    |                Boundaries:                
-------------------------------------------------------------------
   Limits received:   |   X =    |  X type  |   Y =    |  Y type  
-------------------------------------------------------------------
1)        x < y       |    y     | function |    x     | function 
2)        y < 5*x     |   y/5    | function |   5*x    | function 
3)        2 < x+2*y   | 2 - 2*y  | function | 1 - x/2  | function 
4)    x+2*y < 4       | 4 - 2*y  | function | 2 - x/2  | function 
-------------------------------------------------------------------
Vertices of the domain are: [(4/3, 4/3), (2/11, 10/11), (2/3, 2/3), (4/11, 20/11)]
Computing the integral... : 100%|#############| 93637/93637 [00:02<00:00, 40238.72 points checked/s]
            Integral value:  0.172032
                       Min:  0.171623
                       Max:  0.172442
```
![output](https://github.com/user-attachments/assets/85c1d459-57a0-401c-b13d-19cd8e25b7f4)

More datailed explanation is provided within the function docstrings.

## monte_carlo_integration

Straight-forward implementation of MCI. The goal is to demonstrate all the steps of MCI
from taking an arbitrary density function, validating it over the domain, sampling points,
finding MC estimator and comparing to the Newton's-Côtes result.

Visualization and animation are also possible.

Example
```python
def f(x):
  return np.sin(x) + np.cos(x)

def pdf(x):
  return 0.5 * x + 1 if x < 1 else 2.5 - x

mci(f, pdf, [(0, 2)], n_samples=1000, plot=True, verbose=True)
```
```
Task:
Integrand:                      np.sin(x) + np.cos(x)
Raw PDF:                       0.5 * x + 1 if x < 1 else 2.5 - x
PDF validation required:      True
Sampling function provided:   Not Provided
Number of variables:          1
Boundaries of integration:    [0, 2]
Infinite domain:              False
Number of samples:            1000
Plotting option:              True
Animation:                    False

Checking PDF:
PDF minimum:                  0.5001605215288005
PDF integrates to:            2.25
  it will be multiplied by:   0.444

Rejection Sampling from PDF:
Proposal (q(x)):              uniform
pdf(x) / q(x) upper bound:    0.67
Collecting draws from PDF: 100%|██████████| 1000/1000 [00:00<00:00, 239976.20it/s]
Acceptance rate:              0.75
Ranges:                       ['(8.62e-05, 2)']
Mean(s):                      [0.90408331]
Variance(s):                  [0.2872578]

Monte Carlo estimation:
Computing MC estimator: 100%|██████████| 1000/1000 [00:00<00:00, 412378.72it/s]
Number of omitted samples:    0
Variance:                     4.1791e-05
SD:                           0.0064646
The final estimation:         2.33
Confidence interval:          (2.3136 ... 2.3389) p=0.05
Reference value               2.3254
Error:                        0.00080659
```
![3](https://github.com/user-attachments/assets/044db8f1-5067-4d10-82fb-9dccc083a6c2)

## Common dependecies

- python=3.12.2
- numpy=1.26.4
- scipy=1.13.1
- pandas=2.2.2
- matplotlib=3.8.4
- tqdm=4.66.4
- sympy=1.13.3
- regex=2.5.135
- unicodedata=15.0.0
- wordfreq=3.7.0
- spacy=3.7.6
- spellchecker=0.8.0
- bs4=4.12.3

## Contributions

I would appreciate any comments or contributions
