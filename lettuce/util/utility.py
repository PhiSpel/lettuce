import inspect as _inspect
import torch as _torch

__all__ = ['get_subclasses',
           'LettuceException',
           'LettuceWarning',
           'InefficientCodeWarning',
           'ExperimentalWarning'
    , 'torch_gradient',
           'grid_fine_to_coarse',
           'torch_jacobi',
           'append_axes']


def get_subclasses(cls, module):
    for name, obj in _inspect.getmembers(module):
        if hasattr(obj, "__bases__") and cls in obj.__bases__:
            yield obj


class LettuceException(Exception):
    pass


class LettuceWarning(UserWarning):
    pass


class InefficientCodeWarning(LettuceWarning):
    pass


class ExperimentalWarning(LettuceWarning):
    pass


def torch_gradient(f, dx=1, order=2):
    """
    Function to calculate the first derivative of tensors.
    Orders O(h²); O(h⁴); O(h⁶) are implemented.

    Notes
    -----
    See [1]_. The function only works for periodic domains

    References
    ----------
    .. [1]  Fornberg B. (1988) Generation of Finite Difference Formulas on
        Arbitrarily Spaced Grids,
        Mathematics of Computation 51, no. 184 : 699-706.
        `PDF <http://www.ams.org/journals/mcom/1988-51-184/
        S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_.
    """
    dim = f.ndim
    weights = {
        2: [-1 / 2, 1 / 2, 0, 0, 0, 0],
        4: [1 / 12, -2 / 3, 2 / 3, -1 / 12, 0, 0],
        6: [-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60],
    }
    weight = weights.get(order)
    if dim == 2:
        dims = (0, 1)
        stencil = {
            2: [[[1, 0], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 1], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]]],
            4: [[[2, 0], [1, 0], [-1, 0], [-2, 0], [0, 0], [0, 0]],
                [[0, 2], [0, 1], [0, -1], [0, -2], [0, 0], [0, 0]]],
            6: [[[3, 0], [2, 0], [1, 0], [-1, 0], [-2, 0], [-3, 0]],
                [[0, 3], [0, 2], [0, 1], [0, -1], [0, -2], [0, -3]]],
        }
        shift = stencil.get(order)
    elif dim == 3:
        dims = (0, 1, 2)
        stencil = {
            2: [[[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            4: [[[2, 0, 0], [1, 0, 0], [-1, 0, 0], [-2, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 0], [0, 1, 0], [0, -1, 0], [0, -2, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 2], [0, 0, 1], [0, 0, -1], [0, 0, -2], [0, 0, 0], [0, 0, 0]]],
            6: [[[3, 0, 0], [2, 0, 0], [1, 0, 0], [-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
                [[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, -1, 0], [0, -2, 0], [0, -3, 0]],
                [[0, 0, 3], [0, 0, 2], [0, 0, 1], [0, 0, -1], [0, 0, -2], [0, 0, -3]]]
        }
        shift = stencil.get(order)
    else:
        raise LettuceException("Invalid dimension!")
    with _torch.no_grad():
        out = _torch.cat(dim * [f[None, ...]])
        for i in range(dim):
            out[i, ...] = (
                                  weight[0] * f.roll(shifts=shift[i][0], dims=dims) +
                                  weight[1] * f.roll(shifts=shift[i][1], dims=dims) +
                                  weight[2] * f.roll(shifts=shift[i][2], dims=dims) +
                                  weight[3] * f.roll(shifts=shift[i][3], dims=dims) +
                                  weight[4] * f.roll(shifts=shift[i][4], dims=dims) +
                                  weight[5] * f.roll(shifts=shift[i][5], dims=dims)
                          ) * _torch.tensor(1.0 / dx, dtype=f.dtype, device=f.device)
    return out


def grid_fine_to_coarse(flow: 'Flow', f_fine, tau_fine, tau_coarse):
    if f_fine.shape.__len__() == 3:
        f_eq = flow.equilibrium(flow,
                                rho=flow.rho(f_fine[:, ::2, ::2]),
                                u=flow.u(f_fine[:, ::2, ::2]))
        f_neq = f_fine[:, ::2, ::2] - f_eq
    elif f_fine.shape.__len__() == 4:
        f_eq = flow.equilibrium(flow,
                                rho=flow.rho(f_fine[:, ::2, ::2, ::2]),
                                u=flow.u(f_fine[:, ::2, ::2, ::2]))
        f_neq = f_fine[:, ::2, ::2, ::2] - f_eq
    else:
        raise LettuceException("Invalid dimension!")
    f_coarse = f_eq + 2 * tau_coarse / tau_fine * f_neq
    return f_coarse


def torch_jacobi(f, p, dx, dim, tol_abs=1e-10, max_num_steps=100000):
    """Jacobi solver for the Poisson pressure equation"""

    ## transform to torch.tensor
    # p = torch.tensor(p, device=device, dtype=torch.double)
    # dx = torch.tensor(dx, device=device, dtype=torch.double)
    error, it = 1, 0
    while error > tol_abs and it < max_num_steps:
        it += 1
        if dim == 2:
            # Difference quotient for second derivative O(h²) for index i=0,1
            p = (f * (dx ** 2) - (p.roll(shifts=1, dims=0)
                                  + p.roll(shifts=1, dims=1)
                                  + p.roll(shifts=-1, dims=0)
                                  + p.roll(shifts=-1, dims=1))) * -1 / 4
            residuum = f - (p.roll(shifts=1, dims=0)
                            + p.roll(shifts=1, dims=1)
                            + p.roll(shifts=-1, dims=0)
                            + p.roll(shifts=-1, dims=1)
                            - 4 * p) / (dx ** 2)
        if dim == 3:
            # Difference quotient for second derivative O(h²) for index i=0,1,2
            p = (f * (dx ** 2) - (p.roll(shifts=1, dims=0)
                                  + p.roll(shifts=1, dims=1)
                                  + p.roll(shifts=1, dims=2)
                                  + p.roll(shifts=-1, dims=0)
                                  + p.roll(shifts=-1, dims=1)
                                  + p.roll(shifts=-1, dims=2))) * -1 / 6
            residuum = f - (p.roll(shifts=1, dims=0)
                            + p.roll(shifts=1, dims=1)
                            + p.roll(shifts=1, dims=2)
                            + p.roll(shifts=-1, dims=0)
                            + p.roll(shifts=-1, dims=1)
                            + p.roll(shifts=-1, dims=2)
                            - 6 * p) / (dx ** 2)
        # Error is defined as the mean value of the residuum
        error = _torch.mean(residuum ** 2)
    return p


def append_axes(array, n):
    index = (Ellipsis,) + (None,) * n
    return array[index]
