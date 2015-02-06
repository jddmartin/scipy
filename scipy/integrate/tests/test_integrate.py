# Authors: Nils Wagner, Ed Schofield, Pauli Virtanen, John Travers
"""
Tests for numerical integration.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
                   allclose)

from scipy._lib.six import xrange

from numpy.testing import (
    assert_, TestCase, run_module_suite, assert_array_almost_equal,
    assert_raises, assert_allclose, assert_array_equal, assert_equal)
from scipy.integrate import odeint, ode, complex_ode, dense_dop

#------------------------------------------------------------------------------
# Test ODE integrators
#------------------------------------------------------------------------------


class TestOdeint(TestCase):
    # Check integrate.odeint
    def _do_problem(self, problem):
        t = arange(0.0, problem.stop_t, 0.05)
        z, infodict = odeint(problem.f, problem.z0, t, full_output=True)
        assert_(problem.verify(z, t))

    def test_odeint(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem)


class TestODEClass(TestCase):

    ode_class = None   # Set in subclass.

    def _do_problem(self, problem, integrator, method='adams'):

        # ode has callback arguments in different order than odeint
        f = lambda t, z: problem.f(z, t)
        jac = None
        if hasattr(problem, 'jac'):
            jac = lambda t, z: problem.jac(z, t)

        integrator_params = {}
        if problem.lband is not None or problem.uband is not None:
            integrator_params['uband'] = problem.uband
            integrator_params['lband'] = problem.lband

        ig = self.ode_class(f, jac)
        ig.set_integrator(integrator,
                          atol=problem.atol/10,
                          rtol=problem.rtol/10,
                          method=method,
                          **integrator_params)

        ig.set_initial_value(problem.z0, t=0.0)
        z = ig.integrate(problem.stop_t)

        assert_array_equal(z, ig.y)
        assert_(ig.successful(), (problem, method))
        assert_(problem.verify(array([z]), problem.stop_t), (problem, method))


class TestOde(TestODEClass):

    ode_class = ode

    def test_vode(self):
        # Check the vode solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if not problem.stiff:
                self._do_problem(problem, 'vode', 'adams')
            self._do_problem(problem, 'vode', 'bdf')

    def test_zvode(self):
        # Check the zvode solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if not problem.stiff:
                self._do_problem(problem, 'zvode', 'adams')
            self._do_problem(problem, 'zvode', 'bdf')

    def test_lsoda(self):
        # Check the lsoda solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem, 'lsoda')

    def test_dopri5(self):
        # Check the dopri5 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dopri5')

    def test_dop853(self):
        # Check the dop853 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dop853')

    def test_concurrent_fail(self):
        for sol in ('vode', 'zvode', 'lsoda'):
            f = lambda t, y: 1.0

            r = ode(f).set_integrator(sol)
            r.set_initial_value(0, 0)

            r2 = ode(f).set_integrator(sol)
            r2.set_initial_value(0, 0)

            r.integrate(r.t + 0.1)
            r2.integrate(r2.t + 0.1)

            assert_raises(RuntimeError, r.integrate, r.t + 0.1)

    def test_concurrent_ok(self):
        f = lambda t, y: 1.0

        for k in xrange(3):
            for sol in ('vode', 'zvode', 'lsoda', 'dopri5', 'dop853'):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)

                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r2.integrate(r2.t + 0.1)

                assert_allclose(r.y, 0.1)
                assert_allclose(r2.y, 0.2)

            for sol in ('dopri5', 'dop853'):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)

                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                r.integrate(r.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)

                assert_allclose(r.y, 0.3)
                assert_allclose(r2.y, 0.2)


class TestComplexOde(TestODEClass):

    ode_class = complex_ode

    def test_vode(self):
        # Check the vode solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if not problem.stiff:
                self._do_problem(problem, 'vode', 'adams')
            else:
                self._do_problem(problem, 'vode', 'bdf')

    def test_lsoda(self):
        # Check the lsoda solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            self._do_problem(problem, 'lsoda')

    def test_dopri5(self):
        # Check the dopri5 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dopri5')

    def test_dop853(self):
        # Check the dop853 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dop853')


class TestSolout(TestCase):

    # Docstrings for .set_solout in both ode and complex_ode should be the same
    def test_docstrings(self,):
        assert_(ode.set_solout.__doc__ == complex_ode.set_solout.__doc__)
    
    # Check integrate.ode correctly handles solout for dopri5 and dop853
    def _run_solout_test(self, integrator):
        # Check correct usage of solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]

        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)

    def test_solout(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_test(integrator)

    def _run_solout_break_test(self, integrator):
        # Check correct usage of stopping via solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())
            if t > tend/2.0:
                return -1

        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]

        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_(ts[-1] > tend/2.0)
        assert_(ts[-1] < tend)

    def test_solout_break(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_break_test(integrator)

    def _run_dense_chaining_test(self, integrator):
        # test ability to "chain" ode methods:
        class SolOut(object):
            def __init__(self):
                self.solout_called = 0

            def solout(self, *args):
                self.solout_called += 1

        def f(t, y):
            return -y

        aSolOut = SolOut()
        atol = 1.0e-12
        tend = 1.0
        yf = ode(f).set_integrator(integrator, atol=atol, rtol=atol)\
                   .set_solout(aSolOut.solout)\
                   .set_initial_value(1.0).integrate(tend)
        expected_answer = exp(-tend)
        assert_(abs(yf[0] - expected_answer) < atol)
        assert_(aSolOut.solout_called > 1)

    def test_dense_chaining(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_dense_chaining_test(integrator)

    def _run_solout_initial_condition_ordering_test(self, integrator,
                                                    correct_ordering):
        # test related to: https://github.com/scipy/scipy/issues/4118
        def f(t, y):  # Exponential decay.
            return -y

        def solout(t, y):
            if y[0] < 0.5: 
                return -1
            return 0

        y_initial = 1
        t_initial = 0
        r = ode(f).set_integrator(integrator)  # Integrator that supports solout
        if correct_ordering:
            r.set_solout(solout)
            r.set_initial_value(y_initial, t_initial)
        else:  # should raise an exception
            r.set_initial_value(y_initial, t_initial)
            r.set_solout(solout)
        assert_(r.integrate(5)[0] > 0.4)  # make sure we stop before t=5

    def test_solout_initial_condition_ordering(self):
        for integrator in ('dopri5', 'dop853'):
            assert_raises(RuntimeError, 
                          self._run_solout_initial_condition_ordering_test, 
                          integrator,
                          False)
            self._run_solout_initial_condition_ordering_test(integrator, True)

    def _run_test_integrate_arenstorf_ode(self, integrator, tolerance):
        # Check dense output for Arenstorf system.
        # An introductory discussion is given on pages 129-131 of 
        # Hairer et al.'s 
        # "Solving Ordinary Differential Equations, Nonstiff Problems",
        # Second Revised Edition, Springer, 1993.
        # This code is modelled after "Driver for the code DORPI5" in 
        # the Appendix of this book, available at: 
        # http://www.unige.ch/~hairer/prog/nonstiff/dr_dopri5.f

        class SoloutWrapper(object):
            # dense solution points from output of Fortran code
            pretabulated_solution = (
                (2.00, -5.7987814108e-01, 6.0907752507e-01),
                (4.00, -1.9833352699e-01, 1.1376380857e+00),
                (6.00, -4.7357439430e-01, 2.2390681178e-01),
                (8.00, -1.1745533505e+00, -2.7594669824e-01),
                (10.00, -8.3980734662e-01, 4.4683022680e-01),
                (12.00, 1.3147124683e-02, -8.3857514994e-01),
                (14.00, -6.0311295041e-01, -9.9125980314e-01),
                (16.00, 2.4271109988e-01, -3.8999488331e-01),
            )

            def __init__(self, tolerance):
                self.tolerance = tolerance

            def solout(self, nr, told, t, v, con_view, icomp):
                if nr == 1:  # initial conditions:
                    return 0
                # check to see if we can compare any of our pretabulated
                # solution within the interval told to t:
                for tab_t, a, b in self.pretabulated_solution:
                    if ((told <= tab_t) and (tab_t <= t)):
                        dense = dense_dop(tab_t, told, t, con_view)
                        assert_(abs(a-dense[0]) < tolerance)
                        assert_(abs(b-dense[1]) < tolerance)

        def f_arenstorf(x, y, rpar):
            """The Arenstorf system of differential equations.
            """
            amu, amup = rpar
            r1 = (y[0]+amu)**2+y[1]**2
            r1 = r1*sqrt(r1)
            r2 = (y[0]-amup)**2+y[1]**2
            r2 = r2*sqrt(r2)
            f2 = y[0]+2*y[3]-amup*(y[0]+amu)/r1-amu*(y[0]-amup)/r2
            f3 = y[1]-2*y[2]-amup*y[1]/r1-amu*y[1]/r2
            return [y[2], y[3], f2, f3]

        # parameters for differential equation system:
        rpar = zeros((2,), float)
        rpar[0] = 0.012277471
        rpar[1] = 1.0-rpar[0]

        # initial conditions, and length of time to integrate:
        x0 = 0.0
        y0 = [0.994, 0.0, 0.0, -2.00158510637908252240537862224]
        xend = 17.0652165601579625588917206249

        # desired tolerances:
        itol = 0
        rtol = 1.0e-7
        atol = rtol
        ig = ode(f_arenstorf).set_integrator('dopri5', atol=atol, rtol=rtol)
        aSoloutWrapper = SoloutWrapper(tolerance)
        ig.set_solout(aSoloutWrapper.solout, dense_components=(0, 1,))
        ig.set_initial_value(y0, x0).set_f_params(rpar)
        ret = ig.integrate(xend)

    def test_integrate_arenstorf_ode(self):
        # The tolerances chosen are just below threshold of failure:
        for (integrator, tolerance) in (('dopri5', 1e-10),
                                        ('dop853', 1e-10)):
            self._run_test_integrate_arenstorf_ode(integrator, tolerance)
            
class TestComplexSolout(TestCase):
    # Check integrate.ode correctly handles solout for dopri5 and dop853
    def _run_solout_test(self, integrator):
        # Check correct usage of solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 20.0
        y0 = [0.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            return [1.0/(t - 10.0 - 1j)]

        ig = complex_ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)

    def test_solout(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_test(integrator)

    def _run_solout_break_test(self, integrator):
        # Check correct usage of stopping via solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 20.0
        y0 = [0.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())
            if t > tend/2.0:
                return -1

        def rhs(t, y):
            return [1.0/(t - 10.0 - 1j)]

        ig = complex_ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_(ts[-1] > tend/2.0)
        assert_(ts[-1] < tend)

    def test_solout_break(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_break_test(integrator)

    def _run_solout_dense_output_ordering_test(self, integrator, 
                                               dense_components):
        # various specification of `dense_components` will test 
        # dense output correctness and error handling.
        def odeRHS(t, y):  # very simple system with exact solution
            return [1.0j*ay for ay in y]

        class ExactDenseSolution(object):
            def __init__(self, initial_conditions, dense_components):
                self.initial_conditions = initial_conditions
                self.dense_components = dense_components
            
            def solution(self, t):
                all_components_exact = [initial_y*exp(1.0j*t) 
                                        for initial_y in 
                                        self.initial_conditions]
                return [all_components_exact[i] for i in dense_components]

        # just below threshold of failure:
        atol = 1.0e-15
        safety_factor = 10.0
        
        class SoloutWrapper(object):
            def __init__(self, initial_condition, aExactDenseSolution):
                self.initial_condition = initial_condition
                self.aExactDenseSolution = aExactDenseSolution

            def solout(self, nr, told, t, v, con_view, icomp):
                if nr > 1:
                    for tdense in np.linspace(told, t, 10):
                        y_interp = dense_dop(tdense, told, t, con_view)
                        y_exact = self.aExactDenseSolution.solution(tdense)
                        diff = [ay_interp-ay_exact for (ay_interp, ay_exact) 
                                in zip(y_interp, y_exact)]
                        for adiff in diff:
                            assert_(abs(adiff.real) < atol*safety_factor)
                            assert_(abs(adiff.imag) < atol*safety_factor)

        initial_condition = [1.0, 1.0j, -1.0]  
        aExactDenseSolution = ExactDenseSolution(initial_condition, 
                                                 dense_components)
        aSoloutWrapper = SoloutWrapper(initial_condition, aExactDenseSolution)
        ig = complex_ode(odeRHS).set_integrator(integrator, atol=atol,
                                                rtol=atol, nsteps=10000)
        ig.set_solout(aSoloutWrapper.solout, dense_components=dense_components)
        ig.set_initial_value(initial_condition, 0.0)
        ig.integrate(pi)
        
    def test_dense_interpolation(self):
        # check that various ways of specifying the dense components
        # required are handled correctly:
        for integrator in ('dopri5', 'dop853'):
            for dense_components in ((0,), (1,), (2,), (0, 1,), (0, 2,), 
                                     (1, 2), (1, 0,), (2, 1,), (0, 1, 2)):
                self._run_solout_dense_output_ordering_test(integrator,
                                                            dense_components)

            dense_components = (1, 2, 3,)  # index out of range
            assert_raises(ValueError, 
                          self._run_solout_dense_output_ordering_test, 
                          integrator, dense_components)

            dense_components = (0, -1,)  # index out of range
            assert_raises(ValueError, 
                          self._run_solout_dense_output_ordering_test, 
                          integrator, dense_components)

            dense_components = (2, 1, 0,)  # wrong ordering for all components
            assert_raises(ValueError, 
                          self._run_solout_dense_output_ordering_test, 
                          integrator, dense_components)


#------------------------------------------------------------------------------
# Test problems
#------------------------------------------------------------------------------


class ODE:
    """
    ODE problem
    """
    stiff = False
    cmplx = False
    stop_t = 1
    z0 = []

    lband = None
    uband = None

    atol = 1e-6
    rtol = 1e-5


class SimpleOscillator(ODE):
    r"""
    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    """
    stop_t = 1 + 0.09
    z0 = array([1.0, 0.1], float)

    k = 4.0
    m = 1.0

    def f(self, z, t):
        tmp = zeros((2, 2), float)
        tmp[0, 1] = 1.0
        tmp[1, 0] = -self.k / self.m
        return dot(tmp, z)

    def verify(self, zs, t):
        omega = sqrt(self.k / self.m)
        u = self.z0[0]*cos(omega*t) + self.z0[1]*sin(omega*t)/omega
        return allclose(u, zs[:, 0], atol=self.atol, rtol=self.rtol)


class ComplexExp(ODE):
    r"""The equation :lm:`\dot u = i u`"""
    stop_t = 1.23*pi
    z0 = exp([1j, 2j, 3j, 4j, 5j])
    cmplx = True

    def f(self, z, t):
        return 1j*z

    def jac(self, z, t):
        return 1j*eye(5)

    def verify(self, zs, t):
        u = self.z0 * exp(1j*t)
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)


class Pi(ODE):
    r"""Integrate 1/(t + 1j) from t=-10 to t=10"""
    stop_t = 20
    z0 = [0]
    cmplx = True

    def f(self, z, t):
        return array([1./(t - 10 + 1j)])

    def verify(self, zs, t):
        u = -2j * np.arctan(10)
        return allclose(u, zs[-1, :], atol=self.atol, rtol=self.rtol)


class CoupledDecay(ODE):
    r"""
    3 coupled decays suited for banded treatment
    (banded mode makes it necessary when N>>3)
    """

    stiff = True
    stop_t = 0.5
    z0 = [5.0, 7.0, 13.0]
    lband = 1
    uband = 0

    lmbd = [0.17, 0.23, 0.29]  # fictious decay constants

    def f(self, z, t):
        lmbd = self.lmbd
        return np.array([-lmbd[0]*z[0],
                         -lmbd[1]*z[1] + lmbd[0]*z[0],
                         -lmbd[2]*z[2] + lmbd[1]*z[1]])

    def jac(self, z, t):
        # The full Jacobian is
        #
        #    [-lmbd[0]      0         0   ]
        #    [ lmbd[0]  -lmbd[1]      0   ]
        #    [    0      lmbd[1]  -lmbd[2]]
        #
        # The lower and upper bandwidths are lband=1 and uband=0, resp.
        # The representation of this array in packed format is
        #
        #    [-lmbd[0]  -lmbd[1]  -lmbd[2]]
        #    [ lmbd[0]   lmbd[1]      0   ]

        lmbd = self.lmbd
        j = np.zeros((self.lband + self.uband + 1, 3), order='F')

        def set_j(ri, ci, val):
            j[self.uband + ri - ci, ci] = val
        set_j(0, 0, -lmbd[0])
        set_j(1, 0, lmbd[0])
        set_j(1, 1, -lmbd[1])
        set_j(2, 1, lmbd[1])
        set_j(2, 2, -lmbd[2])
        return j

    def verify(self, zs, t):
        # Formulae derived by hand
        lmbd = np.array(self.lmbd)
        d10 = lmbd[1] - lmbd[0]
        d21 = lmbd[2] - lmbd[1]
        d20 = lmbd[2] - lmbd[0]
        e0 = np.exp(-lmbd[0] * t)
        e1 = np.exp(-lmbd[1] * t)
        e2 = np.exp(-lmbd[2] * t)
        u = np.vstack((
            self.z0[0] * e0,
            self.z0[1] * e1 + self.z0[0] * lmbd[0] / d10 * (e0 - e1),
            self.z0[2] * e2 + self.z0[1] * lmbd[1] / d21 * (e1 - e2) +
            lmbd[1] * lmbd[0] * self.z0[0] / d10 *
            (1 / d20 * (e0 - e2) - 1 / d21 * (e1 - e2)))).transpose()
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)


PROBLEMS = [SimpleOscillator, ComplexExp, Pi, CoupledDecay]

#------------------------------------------------------------------------------


def f(t, x):
    dxdt = [x[1], -x[0]]
    return dxdt


def jac(t, x):
    j = array([[0.0, 1.0],
               [-1.0, 0.0]])
    return j


def f1(t, x, omega):
    dxdt = [omega*x[1], -omega*x[0]]
    return dxdt


def jac1(t, x, omega):
    j = array([[0.0, omega],
               [-omega, 0.0]])
    return j


def f2(t, x, omega1, omega2):
    dxdt = [omega1*x[1], -omega2*x[0]]
    return dxdt


def jac2(t, x, omega1, omega2):
    j = array([[0.0, omega1],
               [-omega2, 0.0]])
    return j


def fv(t, x, omega):
    dxdt = [omega[0]*x[1], -omega[1]*x[0]]
    return dxdt


def jacv(t, x, omega):
    j = array([[0.0, omega[0]],
               [-omega[1], 0.0]])
    return j


class ODECheckParameterUse(object):
    """Call an ode-class solver with several cases of parameter use."""

    # This class is intentionally not a TestCase subclass.
    # solver_name must be set before tests can be run with this class.

    # Set these in subclasses.
    solver_name = ''
    solver_uses_jac = False

    def _get_solver(self, f, jac):
        solver = ode(f, jac)
        if self.solver_uses_jac:
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7,
                                  with_jacobian=self.solver_uses_jac)
        else:
            # XXX Shouldn't set_integrator *always* accept the keyword arg
            # 'with_jacobian', and perhaps raise an exception if it is set
            # to True if the solver can't actually use it?
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7)
        return solver

    def _check_solver(self, solver):
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        solver.integrate(pi)
        assert_array_almost_equal(solver.y, [-1.0, 0.0])

    def test_no_params(self):
        solver = self._get_solver(f, jac)
        self._check_solver(solver)

    def test_one_scalar_param(self):
        solver = self._get_solver(f1, jac1)
        omega = 1.0
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    def test_two_scalar_params(self):
        solver = self._get_solver(f2, jac2)
        omega1 = 1.0
        omega2 = 1.0
        solver.set_f_params(omega1, omega2)
        if self.solver_uses_jac:
            solver.set_jac_params(omega1, omega2)
        self._check_solver(solver)

    def test_vector_param(self):
        solver = self._get_solver(fv, jacv)
        omega = [1.0, 1.0]
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)


class DOPRI5CheckParameterUse(ODECheckParameterUse, TestCase):
    solver_name = 'dopri5'
    solver_uses_jac = False


class DOP853CheckParameterUse(ODECheckParameterUse, TestCase):
    solver_name = 'dop853'
    solver_uses_jac = False


class VODECheckParameterUse(ODECheckParameterUse, TestCase):
    solver_name = 'vode'
    solver_uses_jac = True


class ZVODECheckParameterUse(ODECheckParameterUse, TestCase):
    solver_name = 'zvode'
    solver_uses_jac = True


class LSODACheckParameterUse(ODECheckParameterUse, TestCase):
    solver_name = 'lsoda'
    solver_uses_jac = True


def test_odeint_trivial_time():
    # Test that odeint succeeds when given a single time point
    # and full_output=True.  This is a regression test for gh-4282.
    y0 = 1
    t = [0]
    y, info = odeint(lambda y, t: -y, y0, t, full_output=True)
    assert_array_equal(y, np.array([[y0]]))


def test_odeint_banded_jacobian():
    # Test the use of the `Dfun`, `ml` and `mu` options of odeint.

    def func(y, t, c):
        return c.dot(y)

    def jac(y, t, c):
        return c

    def jac_transpose(y, t, c):
        return c.T.copy(order='C')

    def bjac_rows(y, t, c):
        jac = np.row_stack((np.r_[0, np.diag(c, 1)],
                            np.diag(c),
                            np.r_[np.diag(c, -1), 0],
                            np.r_[np.diag(c, -2), 0, 0]))
        return jac

    def bjac_cols(y, t, c):
        return bjac_rows(y, t, c).T.copy(order='C')

    c = array([[-205, 0.01, 0.00, 0.0],
               [0.1, -2.50, 0.02, 0.0],
               [1e-3, 0.01, -2.0, 0.01],
               [0.00, 0.00, 0.1, -1.0]])

    y0 = np.ones(4)
    t = np.array([0, 5, 10, 100])

    # Use the full Jacobian.
    sol1, info1 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=jac)

    # Use the transposed full Jacobian, with col_deriv=True.
    sol2, info2 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=jac_transpose, col_deriv=True)

    # Use the banded Jacobian.
    sol3, info3 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=bjac_rows, ml=2, mu=1)

    # Use the transposed banded Jacobian, with col_deriv=True.
    sol4, info4 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=bjac_cols, ml=2, mu=1, col_deriv=True)

    assert_allclose(sol1, sol2, err_msg="sol1 != sol2")
    assert_allclose(sol1, sol3, atol=1e-12, err_msg="sol1 != sol3")
    assert_allclose(sol3, sol4, err_msg="sol3 != sol4")

    # Verify that the number of jacobian evaluations was the same for the
    # calls of odeint with a full jacobian and with a banded jacobian. This is
    # a regression test--there was a bug in the handling of banded jacobians
    # that resulted in an incorrect jacobian matrix being passed to the LSODA
    # code.  That would cause errors or excessive jacobian evaluations.
    assert_array_equal(info1['nje'], info2['nje'])
    assert_array_equal(info3['nje'], info4['nje'])


def test_odeint_errors():
    def sys1d(x, t):
        return -100*x

    def bad1(x, t):
        return 1.0/0

    def bad2(x, t):
        return "foo"

    def bad_jac1(x, t):
        return 1.0/0

    def bad_jac2(x, t):
        return [["foo"]]

    def sys2d(x, t):
        return [-100*x[0], -0.1*x[1]]

    def sys2d_bad_jac(x, t):
        return [[1.0/0, 0], [0, -0.1]]

    assert_raises(ZeroDivisionError, odeint, bad1, 1.0, [0, 1])
    assert_raises(ValueError, odeint, bad2, 1.0, [0, 1])

    assert_raises(ZeroDivisionError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac1)
    assert_raises(ValueError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac2)

    assert_raises(ZeroDivisionError, odeint, sys2d, [1.0, 1.0], [0, 1],
                  Dfun=sys2d_bad_jac)


def test_odeint_bad_shapes():
    # Tests of some errors that can occur with odeint.

    def badrhs(x, t):
        return [1, -1]

    def sys1(x, t):
        return -100*x

    def badjac(x, t):
        return [[0, 0, 0]]

    # y0 must be at most 1-d.
    bad_y0 = [[0, 0], [0, 0]]
    assert_raises(ValueError, odeint, sys1, bad_y0, [0, 1])

    # t must be at most 1-d.
    bad_t = [[0, 1], [2, 3]]
    assert_raises(ValueError, odeint, sys1, [10.0], bad_t)

    # y0 is 10, but badrhs(x, t) returns [1, -1].
    assert_raises(RuntimeError, odeint, badrhs, 10, [0, 1])

    # shape of array returned by badjac(x, t) is not correct.
    assert_raises(RuntimeError, odeint, sys1, [10, 10], [0, 1], Dfun=badjac)


if __name__ == "__main__":
    run_module_suite()
