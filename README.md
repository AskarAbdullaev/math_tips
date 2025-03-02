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

## Common dependecies

- python=3.12.2
- numpy=1.26.4
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
