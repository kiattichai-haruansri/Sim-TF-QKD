# analysis/decoy_solver.py

from __future__ import annotations

import math
import itertools
import numpy as np
from scipy.optimize import linprog


class DecoyLPSolver:
    """
    Linear Programming Decoy-State Solver
    (Updated with Symmetry Constraints and Truncation Error)
    """

    def __init__(
        self,
        intensities,
        cutoff: int = 4,
    ):
        self.intensities = list(intensities)
        self.cutoff = cutoff
        self.N = cutoff + 1

        self.variables = {}
        self.reverse = {}
        
        k = 0
        for n in range(self.N):
            for m in range(self.N):
                self.variables[(n, m)] = k
                self.reverse[k] = (n, m)
                k += 1

        self.dimension = k

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    @staticmethod
    def poisson_probability(intensity: float, photon: int) -> float:
        return (math.exp(-intensity) * intensity**photon / math.factorial(photon))

    def poisson_vector(self, intensity: float) -> np.ndarray:
        return np.array([self.poisson_probability(intensity, n) for n in range(self.N)], dtype=float)

    def idx(self, n: int, m: int) -> int:
        return self.variables[(n, m)]

    def conditional_probability_matrix(self, mu_a: float, mu_b: float) -> np.ndarray:
        pa = self.poisson_vector(mu_a)
        pb = self.poisson_vector(mu_b)
        return np.outer(pa, pb)

    # -------------------------------------------------
    # LP Constraints
    # -------------------------------------------------

    def build_expectation_constraints(self, expectation_bounds: dict):
        A = []
        b = []

        for (mu_a, mu_b), (lower, upper) in expectation_bounds.items():
            coeff = self.conditional_probability_matrix(mu_a, mu_b).reshape(-1)
            
            # คำนวณความน่าจะเป็นของเทอมที่ถูกตัดทิ้ง (Truncation Error)
            p_rest = 1.0 - np.sum(coeff)
            
            # ปรับ Lower Bound เพื่อชดเชยเทอมที่หายไป
            adjusted_lower = max(0.0, lower - p_rest)

            # coeff @ M <= upper
            A.append(coeff)
            b.append(upper)

            # coeff @ M >= adjusted_lower -> -coeff @ M <= -adjusted_lower
            A.append(-coeff)
            b.append(-adjusted_lower)

        return (
            np.asarray(A, dtype=float),
            np.asarray(b, dtype=float),
        )

    def build_symmetry_constraints(self):
        """
        บังคับสมมาตร: M_nm = M_mn 
        ป้องกันไม่ให้ Solver โยกตัวเลขหนีไปฝั่งใดฝั่งหนึ่ง
        """
        A_eq = []
        b_eq = []
        
        for n in range(self.N):
            for m in range(n + 1, self.N):
                row = np.zeros(self.dimension)
                row[self.idx(n, m)] = 1.0
                row[self.idx(m, n)] = -1.0
                
                A_eq.append(row)
                b_eq.append(0.0)
                
        if not A_eq:
            return None, None
            
        return np.asarray(A_eq, dtype=float), np.asarray(b_eq, dtype=float)

    # -------------------------------------------------

    def variable_bounds(self):
        return [(0.0, 1.0) for _ in range(self.dimension)]

    def objective(self, n: int, m: int, maximize: bool = True):
        c = np.zeros(self.dimension, dtype=float)
        if maximize:
            c[self.idx(n, m)] = -1.0
        else:
            c[self.idx(n, m)] = 1.0
        return c

    def unpack_solution(self, x):
        result = {}
        for idx, value in enumerate(x):
            result[self.reverse[idx]] = value
        return result

    # -------------------------------------------------
    # LP Solver
    # -------------------------------------------------

    def solve_one(self, expectation_bounds: dict, target: tuple[int, int], maximize: bool = True):
        A_ub, b_ub = self.build_expectation_constraints(expectation_bounds)
        A_eq, b_eq = self.build_symmetry_constraints()

        res = linprog(
            c=self.objective(target[0], target[1], maximize=maximize),
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=self.variable_bounds(),
            method="highs",
        )

        if not res.success:
            # ถ้าหาคำตอบไม่ได้เพราะ Noise เยอะเกินไป ให้คืนค่า Worst-case ดิบ
            return {"value": 1.0 if maximize else 0.0}

        value = res.x[self.idx(target[0], target[1])]

        return {
            "target": target,
            "value": value,
            "maximize": maximize,
            "status": res.message,
            "solution": self.unpack_solution(res.x),
        }

    # -------------------------------------------------

    def upper_bound(self, expectation_bounds: dict, n: int, m: int) -> float:
        return self.solve_one(expectation_bounds, target=(n, m), maximize=True)["value"]

    def lower_bound(self, expectation_bounds: dict, n: int, m: int) -> float:
        return self.solve_one(expectation_bounds, target=(n, m), maximize=False)["value"]

    def solve_all(self, expectation_bounds: dict):
        result = {}
        for n in range(self.N):
            for m in range(self.N):
                lower = self.lower_bound(expectation_bounds, n, m)
                upper = self.upper_bound(expectation_bounds, n, m)
                result[(n, m)] = {"lower": lower, "upper": upper}
        return result