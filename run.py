import numpy as np
np.set_printoptions(linewidth=np.inf)
fn = str(input())
print(np.loadtxt(fn).round(3))
