#!/usr/bin/env python3
"""
Model Predictive Contouring Control (MPCC) optimizer

Reference: ETH Zurich MPCC paper (arXiv:1711.07300v1)
- Section III-B: MPCC formulation (Pages 10-13)
- Equation 11: Error definitions (Page 11)
- Equation 14: QP formulation (Page 12)
"""

import numpy as np
import osqp
from scipy import sparse


class MPCCController:
    """
    MPCC optimizer using OSQP QP solver
    
    Decision variables (for N horizon steps):
        z = [x_1, ..., x_N, u_0, ..., u_{N-1}, θ_1, ..., θ_N, v_0, ..., v_{N-1}, s_1, ..., s_N]
    
    Where:
        x_k: State at step k (4 dims: X, Y, φ, v)
        u_k: Input at step k (2 dims: δ, a)
        θ_k: Progress at step k (1 dim: arc-length)
        v_k: Virtual velocity for progress (1 dim)
        s_k: Slack for track boundaries (2 dims: left, right)
    
    Total: N*4 + N*2 + N*1 + N*1 + N*2 = N*10 variables
    """
    
    def __init__(self, vehicle, track, N=10, dt=0.05):
        """
        Initialize MPCC controller
        
        Args:
            vehicle: VehicleModel instance
            track: TrackMap instance
            N: Prediction horizon length
            dt: Timestep (seconds)
        """
        self.vehicle = vehicle
        self.track = track
        self.N = N
        self.dt = dt
        
        # Cost weights (tunable parameters - Equation 14a)
        self.q_c = 10.0       # Contouring error penalty
        self.q_l = 100.0      # Lag error penalty
        self.gamma = 50.0      # Progress reward
        self.R_u = 0.1        # Input regularization
        self.q_slack = 1000.0 # Track boundary slack penalty
        print(f"MPCC Cost weights: q_c={self.q_c}, q_l={self.q_l}, gamma={self.gamma}, R_u={self.R_u}, q_slack={self.q_slack}")
        
        # Constraints (Equation 14g-14h)
        self.delta_max = 0.4    # Maximum steering (rad)
        self.a_max = 3.0        # Maximum acceleration (m/s²)
        self.v_max = 8.0        # Maximum velocity (m/s)
        
        # Warm start storage
        self.x_prev = None
        self.u_prev = None
        self.theta_prev = None
        
        # Setup OSQP solver
        self.solver = osqp.OSQP()
        self.solver_initialized = False
    
    def solve(self, x0, theta0):
        """
        Solve MPCC optimization problem (Equation 14)
        
        Args:
            x0: Initial state [X, Y, φ, v]
            theta0: Initial progress (meters)
            
        Returns:
            (u_opt, x_pred, theta_pred): Optimal first control, predicted states, predicted progress
        """
        # Warm start: guess trajectory
        if self.x_prev is None:
            x_guess = np.tile(x0, (self.N, 1))
            u_guess = np.zeros((self.N, 2))
            
            # FIX: Use current velocity for progress guess
            v_current = max(x0[3], 1.0)  # At least 1 m/s
            theta_guess = theta0 + np.cumsum(np.ones(self.N) * v_current * self.dt)
            # theta_guess = np.mod(theta_guess, self.track.L)  # Wrap around
        else:
            # Shift previous solution (real-time iteration)
            x_guess = np.vstack([self.x_prev[1:], self.x_prev[-1]])
            u_guess = np.vstack([self.u_prev[1:], self.u_prev[-1]])
            theta_guess = np.append(self.theta_prev[1:], self.theta_prev[-1] + 0.1)
        
        # Linearize dynamics at each step (Section III-B4)
        A_list, B_list, g_list = [], [], []
        for k in range(self.N):
            x_ref = x_guess[k]
            u_ref = u_guess[k]
            A, B, g = self.vehicle.linearize(x_ref, u_ref)
            Ad, Bd, gd = self.vehicle.discretize(A, B, g, self.dt)
            A_list.append(Ad)
            B_list.append(Bd)
            g_list.append(gd)
        
        # Build cost matrices (Equation 14a)
        H, q = self._build_cost_matrices(theta_guess)
        
        # Build dynamics constraints (Equation 14c, 14d)
        A_eq, l_eq, u_eq = self._build_dynamics_constraints(A_list, B_list, g_list, x0, theta0)
        
        # Build inequality constraints (Equation 14e-14h)
        A_ineq, l_ineq, u_ineq = self._build_inequality_constraints(theta_guess)
        
        # Combine constraints
        A_qp = sparse.vstack([A_eq, A_ineq])
        l_qp = np.concatenate([l_eq, l_ineq])
        u_qp = np.concatenate([u_eq, u_ineq])
        
        # Setup or update solver
        if not self.solver_initialized:
            self.solver.setup(P=H, q=q, A=A_qp, l=l_qp, u=u_qp,
                            verbose=False, eps_abs=1e-3, eps_rel=1e-3,max_iter=4000, polish=True)
            self.solver_initialized = True
        else:
            self.solver.update(q=q, l=l_qp, u=u_qp)
        
        # Solve QP
        result = self.solver.solve()
        # if result.info.status == 'solved' or result.info.status == 'solved inaccurate':
        #     print(f"\n{'='*70}")
        #     print(f"QP SOLUTION DIAGNOSTIC")
        #     print(f"{'='*70}")
        #     print(f"Solver status: {result.info.status}")
        #     print(f"Iterations: {result.info.iter}")
        #     print(f"Solve time: {result.info.solve_time*1000:.2f}ms")
            
        #     # Extract solution components
        #     x_start = 0
        #     u_start = self.N * 4
        #     theta_start = self.N * 6
        #     v_start = self.N * 7
        #     s_start = self.N * 8
            
        #     x_sol = result.x[x_start:u_start].reshape(self.N, 4)
        #     u_sol = result.x[u_start:theta_start].reshape(self.N, 2)
        #     theta_sol = result.x[theta_start:v_start]
        #     v_virtual_sol = result.x[v_start:s_start]
        #     s_sol = result.x[s_start:].reshape(self.N, 2)
            
        #     # ========================================================================
        #     # 1. DECISION VARIABLES
        #     # ========================================================================
        #     print(f"\n1. DECISION VARIABLES (first 3 steps)")
        #     print(f"{'-'*70}")
        #     for k in range(min(3, self.N)):
        #         print(f"Step {k}:")
        #         print(f"  State x[{k}]:  X={x_sol[k,0]:.4f}, Y={x_sol[k,1]:.4f}, "
        #             f"φ={np.rad2deg(x_sol[k,2]):6.2f}°, v={x_sol[k,3]:.4f}m/s")
        #         print(f"  Input u[{k}]:  δ={np.rad2deg(u_sol[k,0]):6.2f}°, "
        #             f"a={u_sol[k,1]:6.3f}m/s²")
        #         print(f"  Progress θ[{k}]: {theta_sol[k]:.4f}m, "
        #             f"v_virtual={v_virtual_sol[k]:.4f}m/s")
        #         print(f"  Slack s[{k}]:   left={s_sol[k,0]:.4f}m, right={s_sol[k,1]:.4f}m")
            
        #     print(f"\nSummary over horizon:")
        #     print(f"  Virtual velocity: min={v_virtual_sol.min():.3f}, "
        #         f"max={v_virtual_sol.max():.3f}, mean={v_virtual_sol.mean():.3f}m/s")
        #     print(f"  State velocity:   min={x_sol[:,3].min():.3f}, "
        #         f"max={x_sol[:,3].max():.3f}, mean={x_sol[:,3].mean():.3f}m/s")
        #     print(f"  Steering:         min={np.rad2deg(u_sol[:,0].min()):.1f}°, "
        #         f"max={np.rad2deg(u_sol[:,0].max()):.1f}°, mean={np.rad2deg(u_sol[:,0].mean()):.1f}°")
        #     print(f"  Acceleration:     min={u_sol[:,1].min():.2f}, "
        #         f"max={u_sol[:,1].max():.2f}, mean={u_sol[:,1].mean():.2f}m/s²")
        #     print(f"  Progress:         Δθ={theta_sol[-1]-theta_sol[0]:.3f}m "
        #         f"(from {theta_sol[0]:.3f} to {theta_sol[-1]:.3f})")
            
        #     # ========================================================================
        #     # 2. COST BREAKDOWN
        #     # ========================================================================
        #     print(f"\n2. COST BREAKDOWN")
        #     print(f"{'-'*70}")
            
        #     # Compute individual cost components
        #     cost_contouring_total = 0.0
        #     cost_lag_total = 0.0
        #     cost_progress_total = 0.0
        #     cost_input_total = 0.0
        #     cost_slack_total = 0.0
            
        #     print(f"\nPer-step costs (first 3 steps):")
        #     for k in range(min(3, self.N)):
        #         # Get reference
        #         X_ref, Y_ref, Phi = self.track.get_reference(theta_sol[k])
                
        #         # Compute errors
        #         e_c = np.sin(Phi) * (x_sol[k,0] - X_ref) - np.cos(Phi) * (x_sol[k,1] - Y_ref)
        #         e_l = -np.cos(Phi) * (x_sol[k,0] - X_ref) - np.sin(Phi) * (x_sol[k,1] - Y_ref)
                
        #         # Individual costs
        #         cost_c = self.q_c * e_c**2
        #         cost_l = self.q_l * e_l**2
        #         cost_p = -self.gamma * v_virtual_sol[k] * self.dt
        #         cost_u = self.R_u * (u_sol[k,0]**2 + u_sol[k,1]**2)
        #         cost_s = self.q_slack * (s_sol[k,0]**2 + s_sol[k,1]**2)
                
        #         print(f"  k={k}: e_c={e_c:7.4f}m, e_l={e_l:7.4f}m")
        #         print(f"       Contouring: {cost_c:8.4f}  (q_c={self.q_c} × e_c²)")
        #         print(f"       Lag:        {cost_l:8.4f}  (q_l={self.q_l} × e_l²)")
        #         print(f"       Progress:   {cost_p:8.4f}  (-γ={self.gamma} × v_virt × dt)")
        #         print(f"       Input:      {cost_u:8.4f}  (R={self.R_u} × (δ²+a²))")
        #         print(f"       Slack:      {cost_s:8.4f}  (q_slack={self.q_slack} × (s_L²+s_R²))")
        #         print(f"       Step total: {cost_c + cost_l + cost_p + cost_u + cost_s:8.4f}")
                
        #         cost_contouring_total += cost_c
        #         cost_lag_total += cost_l
        #         cost_progress_total += cost_p
        #         cost_input_total += cost_u
        #         cost_slack_total += cost_s
            
        #     # Sum over remaining steps
        #     for k in range(3, self.N):
        #         X_ref, Y_ref, Phi = self.track.get_reference(theta_sol[k])
        #         e_c = np.sin(Phi) * (x_sol[k,0] - X_ref) - np.cos(Phi) * (x_sol[k,1] - Y_ref)
        #         e_l = -np.cos(Phi) * (x_sol[k,0] - X_ref) - np.sin(Phi) * (x_sol[k,1] - Y_ref)
                
        #         cost_contouring_total += self.q_c * e_c**2
        #         cost_lag_total += self.q_l * e_l**2
        #         cost_progress_total += -self.gamma * v_virtual_sol[k] * self.dt
        #         cost_input_total += self.R_u * (u_sol[k,0]**2 + u_sol[k,1]**2)
        #         cost_slack_total += self.q_slack * (s_sol[k,0]**2 + s_sol[k,1]**2)
            
        #     cost_computed = (cost_contouring_total + cost_lag_total + cost_progress_total + 
        #                     cost_input_total + cost_slack_total)
            
        #     print(f"\nTotal costs over horizon:")
        #     print(f"  Contouring error:  {cost_contouring_total:10.4f}  ({cost_contouring_total/cost_computed*100:5.1f}%)")
        #     print(f"  Lag error:         {cost_lag_total:10.4f}  ({cost_lag_total/cost_computed*100:5.1f}%)")
        #     print(f"  Progress reward:   {cost_progress_total:10.4f}  ({cost_progress_total/cost_computed*100:5.1f}%)")
        #     print(f"  Input penalty:     {cost_input_total:10.4f}  ({cost_input_total/cost_computed*100:5.1f}%)")
        #     print(f"  Slack penalty:     {cost_slack_total:10.4f}  ({cost_slack_total/cost_computed*100:5.1f}%)")
        #     print(f"  {'─'*70}")
        #     print(f"  Computed total:    {cost_computed:10.4f}")
        #     print(f"  OSQP objective:    {result.info.obj_val:10.4f}")
        #     print(f"  Match: {'✓' if np.abs(cost_computed - result.info.obj_val) < 0.01 else '✗ MISMATCH!'}")
            
        #     # ========================================================================
        #     # 3. CONSTRAINT SATISFACTION
        #     # ========================================================================
        #     print(f"\n3. CONSTRAINT SATISFACTION")
        #     print(f"{'-'*70}")
            
        #     # Input bounds
        #     delta_violations = np.sum(np.abs(u_sol[:,0]) > self.delta_max)
        #     a_violations = np.sum(np.abs(u_sol[:,1]) > self.a_max)
        #     print(f"Input bounds:")
        #     print(f"  Steering: max_used={np.rad2deg(np.abs(u_sol[:,0]).max()):.2f}°, "
        #         f"limit={np.rad2deg(self.delta_max):.2f}°, violations={delta_violations}")
        #     print(f"  Accel:    max_used={np.abs(u_sol[:,1]).max():.3f}m/s², "
        #         f"limit={self.a_max:.3f}m/s², violations={a_violations}")
            
        #     # Velocity bounds
        #     v_violations = np.sum((x_sol[:,3] < 0) | (x_sol[:,3] > self.v_max))
        #     print(f"\nVelocity bounds:")
        #     print(f"  State v: min={x_sol[:,3].min():.3f}, max={x_sol[:,3].max():.3f}m/s, "
        #         f"limit=[0, {self.v_max}], violations={v_violations}")
        #     print(f"  Virtual v: min={v_virtual_sol.min():.3f}, max={v_virtual_sol.max():.3f}m/s, "
        #         f"limit=[0, {self.v_max}], violations={np.sum((v_virtual_sol < 0) | (v_virtual_sol > self.v_max))}")
            
        #     # Progress bounds
        #     theta_violations = np.sum((theta_sol < 0) | (theta_sol > 2 * self.track.L))
        #     print(f"\nProgress bounds:")
        #     print(f"  θ: min={theta_sol.min():.3f}, max={theta_sol.max():.3f}m, "
        #         f"limit=[0, {self.track.L:.1f}], violations={theta_violations}")
            
        #     # Slack non-negativity
        #     slack_violations = np.sum(s_sol < 0)
        #     print(f"\nSlack non-negativity:")
        #     print(f"  min(s)={s_sol.min():.6f}, violations={slack_violations}")
            
        #     # Track boundaries
        #     print(f"\nTrack boundary violations:")
        #     for k in range(min(3, self.N)):
        #         F, f = self.track.get_halfspace_constraints(theta_sol[k])
        #         X, Y = x_sol[k,0], x_sol[k,1]
                
        #         # Check: F·[X,Y] - s ≤ f
        #         for i in range(2):
        #             lhs = F[i,0] * X + F[i,1] * Y - s_sol[k,i]
        #             rhs = f[i]
        #             violated = lhs > rhs + 1e-6
                    
        #             bound_name = "left" if i == 0 else "right"
        #             print(f"  k={k}, {bound_name}: F·[X,Y]-s={lhs:.4f}, limit={rhs:.4f}, "
        #                 f"violation={max(0, lhs-rhs):.4f} {'✗' if violated else '✓'}")
            
        #     # ========================================================================
        #     # 4. DYNAMICS CONSISTENCY
        #     # ========================================================================
        #     print(f"\n4. DYNAMICS CONSISTENCY CHECK")
        #     print(f"{'-'*70}")
            
        #     print(f"Vehicle dynamics (first 2 steps):")
        #     for k in range(min(2, self.N)):
        #         if k == 0:
        #             x_curr = x0
        #         else:
        #             x_curr = x_sol[k-1]
                
        #         # Forward propagate using dynamics
        #         A, B, g = self.vehicle.linearize(x_curr, u_sol[k])
        #         Ad, Bd, gd = self.vehicle.discretize(A, B, g, self.dt)
        #         x_pred = Ad @ x_curr + Bd @ u_sol[k] + gd
        #         x_actual = x_sol[k]
                
        #         error = np.linalg.norm(x_pred - x_actual)
        #         print(f"  k={k}: predicted={x_pred}, actual={x_actual}")
        #         print(f"       error={error:.6f} {'✓' if error < 1e-3 else '✗ MISMATCH!'}")
            
        #     print(f"\nProgress dynamics:")
        #     for k in range(min(3, self.N)):
        #         if k == 0:
        #             theta_curr = theta0
        #         else:
        #             theta_curr = theta_sol[k-1]
                
        #         theta_pred = theta_curr + v_virtual_sol[k] * self.dt
        #         theta_actual = theta_sol[k]
        #         error = abs(theta_pred - theta_actual)
                
        #         print(f"  k={k}: θ_pred={theta_pred:.4f}, θ_actual={theta_actual:.4f}, "
        #             f"error={error:.6f} {'✓' if error < 1e-4 else '✗'}")
            
        #     # ========================================================================
        #     # 5. DIAGNOSIS
        #     # ========================================================================
        #     print(f"\n5. DIAGNOSIS")
        #     print(f"{'-'*70}")
            
        #     # Check what's dominant in cost
        #     costs = {
        #         'Contouring': cost_contouring_total,
        #         'Lag': cost_lag_total,
        #         'Progress': abs(cost_progress_total),
        #         'Input': cost_input_total,
        #         'Slack': cost_slack_total
        #     }
        #     dominant_cost = max(costs, key=costs.get)
            
        #     print(f"Dominant cost term: {dominant_cost} ({costs[dominant_cost]:.2f})")
            
        #     # Check if optimizer is trying to move
        #     if v_virtual_sol.mean() < 0.1:
        #         print(f"⚠️  Virtual velocity near ZERO → optimizer not trying to make progress!")
        #         print(f"    → Progress reward too weak OR constraints blocking motion")
        #     elif v_virtual_sol.mean() > 1.0:
        #         print(f"✓  Virtual velocity good ({v_virtual_sol.mean():.2f}m/s) → optimizer wants progress")
            
        #     # Check if controls are saturated
        #     if np.abs(u_sol[:,0]).max() > 0.95 * self.delta_max:
        #         print(f"⚠️  Steering SATURATED at {np.rad2deg(np.abs(u_sol[:,0]).max()):.1f}°")
        #         print(f"    → Track boundaries forcing hard turns OR contouring weight too high")
            
        #     if np.abs(u_sol[:,1]).max() > 0.95 * self.a_max:
        #         accel_sign = "BRAKING" if u_sol[0,1] < 0 else "ACCELERATING"
        #         print(f"⚠️  Acceleration SATURATED at {u_sol[0,1]:.2f}m/s² ({accel_sign})")
        #         if u_sol[0,1] < 0:
        #             print(f"    → Optimizer wants to SLOW DOWN!")
        #             print(f"    → Check: Is moving actually reducing cost?")
            
        #     # Check slack usage
        #     if s_sol.max() > 0.01:
        #         print(f"⚠️  Slack variables active (max={s_sol.max():.3f}m)")
        #         print(f"    → Track boundaries being violated → may be too tight")
            
        #     print(f"{'='*70}\n")
        
        # else:
        #     print(f"❌ QP FAILED: {result.info.status}")
        #     print(f"{'='*70}\n")

        self.last_result = result

        z_opt = result.x
        x_opt, u_opt, theta_opt = self._unpack_solution(z_opt)

        v_start = self.N * 7
        v_opt = z_opt[v_start:v_start + self.N]
        self.v_virtual_prev = v_opt  # Store for logging
        
        # === ADD THIS: Extract virtual velocity for logging ===
        v_start = self.N * 7
        v_opt = z_opt[v_start:v_start + self.N]
        self.v_virtual_prev = v_opt  # Store for logger
        
        # Store for warm start
        self.x_prev = x_opt
        self.u_prev = u_opt
        self.theta_prev = theta_opt
        
        if result.info.status != 'solved':
            print(f"⚠️  OSQP status: {result.info.status}")
        

        
        # Return first control
        return u_opt[0], x_opt, theta_opt
    
    def _build_cost_matrices(self, theta_guess):
        """
        Build quadratic cost matrices H and q (Equation 14a)
        
        Cost function:
            Σ [||ê_c||²_{q_c} + ||ê_l||²_{q_l} + ||Δu||²_R - γ·v·dt] + slack penalty
        
        Where:
            ê_c: Linearized contouring error (Equation 11a)
            ê_l: Linearized lag error (Equation 11b)
        
        Returns:
            H: Quadratic cost matrix (sparse, n_vars × n_vars)
            q: Linear cost vector (n_vars,)
        """
        n_vars = self.N * 10
        
        H_data = []
        H_row = []
        H_col = []
        q = np.zeros(n_vars)
        
        # Variable indices in decision vector z
        x_start = 0                # States: x_1, ..., x_N
        u_start = self.N * 4       # Inputs: u_0, ..., u_{N-1}
        theta_start = self.N * 6   # Progress: θ_1, ..., θ_N
        v_start = self.N * 7       # Virtual velocity: v_0, ..., v_{N-1}
        s_start = self.N * 8       # Slack: s_1, ..., s_N
        
        for k in range(self.N):
            # Get reference point on centerline
            theta_k = theta_guess[k]
            X_ref, Y_ref, Phi = self.track.get_reference(theta_k)
            
            # Precompute trigonometric values
            s_phi = np.sin(Phi)  # sin(Φ)
            c_phi = np.cos(Phi)  # cos(Φ)
            
            # ================================================================
            # CONTOURING ERROR COST: q_c·||ê_c||² (Equation 11a)
            # ================================================================
            # ê_c = sin(Φ)·(X - X_ref) - cos(Φ)·(Y - Y_ref)
            # 
            # Quadratic expansion:
            # (ê_c)² = [sin(Φ)·X - cos(Φ)·Y]² - 2·[sin(Φ)·X - cos(Φ)·Y]·[sin(Φ)·X_ref - cos(Φ)·Y_ref] + const
            #
            # Hessian H_c (for X, Y):
            #   H_c = q_c · [ sin²(Φ)           -sin(Φ)·cos(Φ) ]
            #               [ -sin(Φ)·cos(Φ)    cos²(Φ)         ]
            #
            # Linear term g_c (for X, Y):
            #   ref_term = sin(Φ)·X_ref - cos(Φ)·Y_ref
            #   g_c = q_c · [ -2·sin(Φ)·ref_term ]
            #               [  2·cos(Φ)·ref_term ]
            
            ref_term_c = s_phi * X_ref - c_phi * Y_ref
            
            # Hessian for contouring (2×2 matrix for X, Y)
            H_c_XX = self.q_c * s_phi**2
            H_c_XY = -self.q_c * s_phi * c_phi
            H_c_YY = self.q_c * c_phi**2
            
            # Linear term for contouring
            g_c_X = -2.0 * self.q_c * s_phi * ref_term_c
            g_c_Y = 2.0 * self.q_c * c_phi * ref_term_c
            
            # ================================================================
            # LAG ERROR COST: q_l·||ê_l||² (Equation 11b)
            # ================================================================
            # ê_l = -cos(Φ)·(X - X_ref) - sin(Φ)·(Y - Y_ref)
            #
            # Quadratic expansion:
            # (ê_l)² = [cos(Φ)·X + sin(Φ)·Y]² - 2·[cos(Φ)·X + sin(Φ)·Y]·[cos(Φ)·X_ref + sin(Φ)·Y_ref] + const
            #
            # Hessian H_l (for X, Y):
            #   H_l = q_l · [ cos²(Φ)          sin(Φ)·cos(Φ) ]
            #               [ sin(Φ)·cos(Φ)    sin²(Φ)        ]
            #
            # Linear term g_l (for X, Y):
            #   ref_term = cos(Φ)·X_ref + sin(Φ)·Y_ref
            #   g_l = q_l · [ -2·cos(Φ)·ref_term ]
            #               [ -2·sin(Φ)·ref_term ]
            
            ref_term_l = c_phi * X_ref + s_phi * Y_ref
            
            # Hessian for lag (2×2 matrix for X, Y)
            H_l_XX = self.q_l * c_phi**2
            H_l_XY = self.q_l * s_phi * c_phi
            H_l_YY = self.q_l * s_phi**2
            
            # Linear term for lag
            g_l_X = -2.0 * self.q_l * c_phi * ref_term_l
            g_l_Y = -2.0 * self.q_l * s_phi * ref_term_l
            
            # ================================================================
            # COMBINE CONTOURING + LAG ERROR COSTS
            # ================================================================
            
            x_idx = x_start + k * 4  # State x_k = [X, Y, φ, v]
            
            # H[X, X] = H_c[X,X] + H_l[X,X]
            H_data.append(H_c_XX + H_l_XX)
            H_row.append(x_idx)
            H_col.append(x_idx)
            
            # H[X, Y] = H_c[X,Y] + H_l[X,Y]
            H_data.append(H_c_XY + H_l_XY)
            H_row.append(x_idx)
            H_col.append(x_idx + 1)
            
            # H[Y, X] = H_c[Y,X] + H_l[Y,X] (symmetric)
            H_data.append(H_c_XY + H_l_XY)
            H_row.append(x_idx + 1)
            H_col.append(x_idx)
            
            # H[Y, Y] = H_c[Y,Y] + H_l[Y,Y]
            H_data.append(H_c_YY + H_l_YY)
            H_row.append(x_idx + 1)
            H_col.append(x_idx + 1)
            
            # Linear terms q[X], q[Y]
            q[x_idx] += g_c_X + g_l_X
            q[x_idx + 1] += g_c_Y + g_l_Y
            
            # ================================================================
            # INPUT REGULARIZATION: ||Δu||²_R (Equation 14a)
            # ================================================================
            # Penalize rate of change in steering and acceleration
            
            u_idx = u_start + k * 2  # u_k = [δ, a]
            
            # R[δ, δ]
            H_data.append(self.R_u)
            H_row.append(u_idx)
            H_col.append(u_idx)
            
            # R[a, a]
            H_data.append(self.R_u)
            H_row.append(u_idx + 1)
            H_col.append(u_idx + 1)
            
            # ================================================================
            # PROGRESS REWARD: -γ·v_k·dt (Equation 14a)
            # ================================================================
            # Maximize progress by penalizing negative virtual velocity
            
            v_idx = v_start + k
            q[v_idx] = -self.gamma * self.dt
            
            # ================================================================
            # SLACK PENALTY: q_slack·||s||_∞ (Equation 14a)
            # ================================================================
            # High penalty to recover hard constraints when feasible
            
            s_idx = s_start + k * 2  # s_k = [s_left, s_right]
            
            # Penalty for left slack
            H_data.append(self.q_slack)
            H_row.append(s_idx)
            H_col.append(s_idx)
            
            # Penalty for right slack
            H_data.append(self.q_slack)
            H_row.append(s_idx + 1)
            H_col.append(s_idx + 1)
        
        # Build sparse Hessian matrix
        H = sparse.csc_matrix((H_data, (H_row, H_col)), shape=(n_vars, n_vars))
        
        return H, q
    
    def _build_dynamics_constraints(self, A_list, B_list, g_list, x0, theta0):
        """
        Build equality constraints for dynamics (Equation 14c, 14d)
        
        Constraints:
            x_{k+1} = A_k·x_k + B_k·u_k + g_k  (Equation 14c - vehicle dynamics)
            θ_{k+1} = θ_k + v_k·dt              (Equation 14d - progress evolution)
            x_0 = x0, θ_0 = θ0                  (Equation 14b - initial conditions)
        
        In QP form: l_eq ≤ A_eq·z ≤ u_eq
        For equality: l_eq = u_eq = b
        
        Returns:
            A_eq: Constraint matrix (sparse, n_constraints × n_vars)
            l_eq: Lower bound vector (n_constraints,)
            u_eq: Upper bound vector (n_constraints,)
        """
        n_vars = self.N * 10
        n_constraints = self.N * 4 + self.N  # Vehicle dynamics + progress dynamics
        
        rows = []
        cols = []
        data = []
        b = np.zeros(n_constraints)
        
        # Variable indices
        x_start = 0
        u_start = self.N * 4
        theta_start = self.N * 6
        v_start = self.N * 7
        
        constraint_idx = 0
        
        # ================================================================
        # VEHICLE DYNAMICS: x_{k+1} = A_k·x_k + B_k·u_k + g_k (Equation 14c)
        # ================================================================
        # Rearranged: x_{k+1} - A_k·x_k - B_k·u_k = g_k
        
        for k in range(self.N):
            Ad = A_list[k]
            Bd = B_list[k]
            gd = g_list[k]
            
            # Indices for current and next state
            if k == 0:
                x_curr = x0  # Initial condition
            else:
                x_curr_idx = x_start + (k - 1) * 4
            
            x_next_idx = x_start + k * 4
            u_curr_idx = u_start + k * 2
            
            # Build constraint for each state dimension
            for i in range(4):  # [X, Y, φ, v]
                # Coefficient for x_{k+1}[i]: +1
                rows.append(constraint_idx + i)
                cols.append(x_next_idx + i)
                data.append(1.0)
                
                # Coefficients for -A_k·x_k
                if k == 0:
                    # Use x0 directly (initial condition)
                    b[constraint_idx + i] = gd[i] + np.dot(Ad[i, :], x0)
                else:
                    # Add -A_k terms
                    for j in range(4):
                        rows.append(constraint_idx + i)
                        cols.append(x_curr_idx + j)
                        data.append(-Ad[i, j])
                    b[constraint_idx + i] = gd[i]
                
                # Coefficients for -B_k·u_k
                for j in range(2):  # [δ, a]
                    rows.append(constraint_idx + i)
                    cols.append(u_curr_idx + j)
                    data.append(-Bd[i, j])
            
            constraint_idx += 4
        
        # ================================================================
        # PROGRESS DYNAMICS: θ_{k+1} = θ_k + v_k·dt (Equation 14d)
        # ================================================================
        # Rearranged: θ_{k+1} - θ_k - v_k·dt = 0
        
        for k in range(self.N):
            if k == 0:
                # First step: θ_1 = θ_0 + v_0·dt
                # Rearranged: θ_1 - v_0·dt = θ_0
                
                theta_next_idx = theta_start  # θ_1
                v_curr_idx = v_start          # v_0
                
                # Coefficient for θ_1: +1
                rows.append(constraint_idx)
                cols.append(theta_next_idx)
                data.append(1.0)
                
                # Coefficient for v_0: -dt
                rows.append(constraint_idx)
                cols.append(v_curr_idx)
                data.append(-self.dt)
                
                # RHS: θ_0 (initial progress)
                b[constraint_idx] = theta0
            else:
                # Subsequent steps: θ_{k+1} = θ_k + v_k·dt
                # Rearranged: θ_{k+1} - θ_k - v_k·dt = 0
                
                theta_next_idx = theta_start + k
                theta_curr_idx = theta_start + k - 1
                v_curr_idx = v_start + k
                
                # Coefficient for θ_{k+1}: +1
                rows.append(constraint_idx)
                cols.append(theta_next_idx)
                data.append(1.0)
                
                # Coefficient for θ_k: -1
                rows.append(constraint_idx)
                cols.append(theta_curr_idx)
                data.append(-1.0)
                
                # Coefficient for v_k: -dt
                rows.append(constraint_idx)
                cols.append(v_curr_idx)
                data.append(-self.dt)
                
                # RHS: 0
                b[constraint_idx] = 0.0
            
            constraint_idx += 1
        
        # Build sparse constraint matrix
        A_eq = sparse.csc_matrix((data, (rows, cols)), shape=(n_constraints, n_vars))
        
        # For OSQP: equality constraints are l_eq = u_eq = b
        l_eq = b
        u_eq = b
        
        return A_eq, l_eq, u_eq
    
    def _build_inequality_constraints(self, theta_guess):
        """
        Build inequality constraints (Equation 14e-14h)
        """
        n_vars = self.N * 10
        
        # Count inequality constraints
        n_input_bounds = self.N * 2      # δ and a bounds
        n_state_v_bounds = self.N        # State velocity v bounds
        n_virtual_v_bounds = self.N      # Virtual velocity bounds
        n_theta_bounds = self.N          # Progress bounds
        n_slack_bounds = self.N * 2      # Slack non-negativity
        n_track_bounds = self.N * 2      # Track boundary constraints
        
        n_constraints = (n_input_bounds + n_state_v_bounds + 
                        n_virtual_v_bounds + n_theta_bounds + n_slack_bounds + n_track_bounds + self.N)
        
        rows = []
        cols = []
        data = []
        l_ineq = []
        u_ineq = []
        
        # Variable indices
        x_start = 0
        u_start = self.N * 4
        theta_start = self.N * 6
        v_start = self.N * 7
        s_start = self.N * 8
        
        constraint_idx = 0
        
        # ================================================================
        # 1. INPUT BOUNDS: -δ_max ≤ δ ≤ δ_max, -a_max ≤ a ≤ a_max
        # ================================================================
        
        for k in range(self.N):
            u_idx = u_start + k * 2  # ← THIS LINE IS CRITICAL!
            
            # Steering constraint: -δ_max ≤ δ_k ≤ δ_max
            rows.append(constraint_idx)
            cols.append(u_idx)
            data.append(1.0)
            l_ineq.append(-self.delta_max)
            u_ineq.append(self.delta_max)
            constraint_idx += 1
            
            # Acceleration constraint: -a_max ≤ a_k ≤ a_max
            rows.append(constraint_idx)
            cols.append(u_idx + 1)
            data.append(1.0)
            l_ineq.append(-self.a_max)
            u_ineq.append(self.a_max)
            constraint_idx += 1
        
        # ================================================================
        # 2. STATE VELOCITY BOUNDS: v_min ≤ v_k ≤ v_max
        # ================================================================
        
        v_min = 0.0  # Minimum forward velocity (m/s)
        
        for k in range(self.N):
            v_state_idx = x_start + k * 4 + 3  # v is 4th element of x_k
            
            rows.append(constraint_idx)
            cols.append(v_state_idx)
            data.append(1.0)
            l_ineq.append(v_min)      # Enforce minimum velocity
            u_ineq.append(self.v_max)
            constraint_idx += 1
        
        # ================================================================
        # 3. VIRTUAL VELOCITY BOUNDS: 0 ≤ v_k ≤ v_max
        # ================================================================
        
        for k in range(self.N):
            v_idx = v_start + k
            
            rows.append(constraint_idx)
            cols.append(v_idx)
            data.append(1.0)
            l_ineq.append(0.0)
            u_ineq.append(self.v_max)
            constraint_idx += 1
        
        # ================================================================
        # 4. PROGRESS BOUNDS: 0 ≤ θ_k ≤ L
        # ================================================================
        
        for k in range(self.N):
            theta_idx = theta_start + k
            
            rows.append(constraint_idx)
            cols.append(theta_idx)
            data.append(1.0)
            l_ineq.append(0.0)
            u_ineq.append(2.0 * self.track.L)
            constraint_idx += 1
        
        # ================================================================
        # 5. SLACK NON-NEGATIVITY: s_k ≥ 0
        # ================================================================
        
        for k in range(self.N):
            s_idx = s_start + k * 2
            
            # Left slack
            rows.append(constraint_idx)
            cols.append(s_idx)
            data.append(1.0)
            l_ineq.append(0.0)
            u_ineq.append(np.inf)
            constraint_idx += 1
            
            # Right slack
            rows.append(constraint_idx)
            cols.append(s_idx + 1)
            data.append(1.0)
            l_ineq.append(0.0)
            u_ineq.append(np.inf)
            constraint_idx += 1

        # Track boundary constraints: F·[X, Y]' - s ≤ f
        for k in range(self.N):
            theta_k = theta_guess[k]
            F, f = self.track.get_halfspace_constraints(theta_k)
            
            x_idx = x_start + k * 4  # [X, Y, φ, v]
            s_idx = s_start + k * 2
            
            for i in range(2):  # Left and right boundaries
                # F[i,0]·X + F[i,1]·Y - s[i] ≤ f[i]
                rows.append(constraint_idx)
                cols.append(x_idx)      # X coefficient
                data.append(F[i, 0])
                
                rows.append(constraint_idx)
                cols.append(x_idx + 1)  # Y coefficient
                data.append(F[i, 1])
                
                rows.append(constraint_idx)
                cols.append(s_idx + i)  # Slack
                data.append(-1.0)
                
                l_ineq.append(-np.inf)
                u_ineq.append(f[i])
                constraint_idx += 1

        # ================================================================
        # 6. VELOCITY COUPLING: v_virtual ≤ v_state
        # ================================================================
        # Virtual progress cannot exceed actual car velocity

        for k in range(self.N):
            v_virtual_idx = v_start + k
            v_state_idx = x_start + k * 4 + 3  # v is 4th element of x_k
            
            # Constraint: v_virtual - v_state ≤ 0
            rows.append(constraint_idx)
            cols.append(v_virtual_idx)
            data.append(1.0)
            
            rows.append(constraint_idx)
            cols.append(v_state_idx)
            data.append(-1.0)
            
            l_ineq.append(-np.inf)
            u_ineq.append(0.0)
            constraint_idx += 1
                
        # Build sparse matrix
        A_ineq = sparse.csc_matrix((data, (rows, cols)), shape=(n_constraints, n_vars))
        l_ineq = np.array(l_ineq)
        u_ineq = np.array(u_ineq)
        
        return A_ineq, l_ineq, u_ineq
    
    def _unpack_solution(self, z):
        """
        Unpack optimization solution into states, inputs, progress
        
        Args:
            z: Decision vector [x_1,...,x_N, u_0,...,u_{N-1}, θ_1,...,θ_N, v_0,...,v_{N-1}, s_1,...,s_N]
        
        Returns:
            x_opt: Predicted states (N × 4)
            u_opt: Optimal inputs (N × 2)
            theta_opt: Predicted progress (N,)
        """
        x_opt = z[0:self.N*4].reshape(self.N, 4)
        u_opt = z[self.N*4:self.N*6].reshape(self.N, 2)
        theta_opt = z[self.N*6:self.N*7]
        
        return x_opt, u_opt, theta_opt