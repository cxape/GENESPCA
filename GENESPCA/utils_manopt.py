import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
from pymanopt.function import autograd
import autograd.numpy as np


def manopt_minimize(A_list, m, d):  # 定义带装饰器的目标函数
    manifold = Stiefel(m, d)
    @pymanopt.function.autograd(manifold)
    def cost(X):
        return np.sum(X[:,i].T @ A_list[i] @ X[:,i] for i in range(d))
    # 优化流程
    problem = Problem(manifold=manifold, cost=cost)
    optimizer = TrustRegions(verbosity=0)
    result = optimizer.run(problem)
    return result.point

