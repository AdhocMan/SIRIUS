def _fermi_entropy(fn, dd):
    import numpy as np
    return -np.sum(fn * np.log(fn + dd * (2 - fn)) +
                   (2 - fn) * np.log(2 - fn + dd * fn))


def fermi_entropy(fn, dd=1e-4):
    """
    Keyword Arguments:
    fn --  occupation numbers
    dd -- regularization parameter
    """
    from ..coefficient_array import CoefficientArray
    import numpy as np
    if isinstance(fn, CoefficientArray):
        out = CoefficientArray(dtype=np.double, ctype=np.array)
        for key, val in fn._data.items():
            out[key] = _fermi_entropy(val, dd)
        return out
    else:
        return _fermi_entropy(fn, dd)


def df_fermi_entropy(fn, dd=1e-4):
    """
    Keyword Arguments:
    fn --  occupation numbers
    dd -- regularization parameter
    """
    from ..coefficient_array import CoefficientArray
    import numpy as np

    if isinstance(fn, CoefficientArray):
        out = CoefficientArray(dtype=np.double, ctype=np.array)
        for key, val in fn._data.items():
            out[key] = _df_fermi_entropy(val, dd)
        return out
    else:
        return _df_fermi_entropy(fn, dd)


def _df_fermi_entropy(fn, dd):
    import numpy as np
    return -fn * (1 - dd) / (fn + dd * (2 - fn)) + (2 - fn) * (-1 + dd) / (
        2 - fn + dd * fn) - np.log(fn + dd *
                                   (2 - fn)) - np.log(2 - fn + dd * fn)


def _constrain_occupancy_gradient(dfn, fn):
    """

    """
    from scipy.optimize import minimize, Bounds
    import numpy as np
    s = 100
    lb = -s * np.ones_like(fn)
    ub = s * np.ones_like(fn)
    ub[fn == 2] = 0
    lb[fn == 0] = 0

    bounds = Bounds(lb, ub)
    x0 = dfn
    res = minimize(
        lambda x: np.linalg.norm(x - dfn),
        x0,
        bounds=bounds,
        constraints={
            'fun': lambda y: np.sum(y),
            "type": "eq"
        })
    y = res['x']
    return y


def constrain_occupancy_gradient(fn):
    from ..coefficient_array import CoefficientArray
    import numpy as np

    if isinstance(fn, CoefficientArray):
        out = CoefficientArray(dtype=np.double, ctype=np.array)
        for key, val in fn._data.items():
            out[key] = _constrain_occupancy_gradient(val)
        return out
    else:
        return _constrain_occupancy_gradient(fn)


class FreeEnergy:
    def __init__(self, energy, temperature):
        self.energy = energy
        self.temperature = temperature

    def __call__(self, cn, fn):
        """
        Keyword Arguments:
        cn   -- Planewave coefficients
        fn   -- occupations numbers
        """
        import numpy as np

        self.energy.kpointset.fn = fn
        E = self.energy(cn)
        S = fermi_entropy(fn)

        omega_k = np.array([
            self.energy.kpointset[k].weight()
            for k in range(len(self.energy.kpointset))
        ])

        return E - self.temperature * np.sum(omega_k * np.array(list(S._data)))

    def grad(self, cn, fn):
        """
        Keyword Arguments:
        self --
        cn   -- planewave coefficients
        fn   -- occupation numbers

        Returns:
        dAdC -- gradient with respect to pw coeffs
        dAdf -- gradient with respect to occupation numbers
        """
        from ..coefficient_array import einsum
        import numpy as np

        # Compute dAdC
        self.energy.kpointset.fn = fn
        dAdC = self.energy.H @ cn

        # Compute dAdf
        # TODO: k-point weights missing?
        dAdfn = np.real(einsum('ij,ij->j', cn.conj(),
                               dAdC)) - self.temperature * df_fermi_entropy(fn)
        return dAdC, dAdfn.flatten(ctype=np.array)
