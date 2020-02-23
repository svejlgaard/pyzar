import numpy as np

def get_lines(z: 'redshift'):
    lines = np.array([1216, 1670, 5006, 1526, 2344, 2374, 2260, 2249, 2383, 2586, 2600, 2026, 2062, 2056, 2066, 1808, 1611])
    lines = (1+z) * lines
    print(np.sort(lines),'\n', len(lines))


get_lines(float(input('What is the redshift? ')))