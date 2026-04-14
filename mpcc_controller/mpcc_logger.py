#!/usr/bin/env python3
"""
UNIFIED MPCC Logger - Single CSV with all data

Logs everything to one file: mpcc_unified_log.csv
- State data (X, Y, φ, v)
- Control inputs (δ, a)
- Errors (e_c, e_l)
- Costs (contouring, lag, progress, input, slack)
- Progress (θ, v_virtual)
- Solver stats (status, iterations, solve time)
- Constraints (violations)
"""

import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import atexit
import signal


class MPCCLogger:
    """
    Single-file MPCC logger with proper resource management
    """
    
    def __init__(self, log_dir='mpcc_logs', enable=True):
        """
        Initialize unified logger
        
        Args:
            log_dir: Directory to save log file
            enable: Enable/disable logging
        """
        self.enable = enable
        self.closed = False
        
        if not self.enable:
            return
        
        # Create log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / f'run_{timestamp}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Register cleanup handlers
        atexit.register(self.close)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Single CSV file
            self.filepath = self.log_dir / 'mpcc_unified_log.csv'
            
            # Open with line buffering
            self.file = open(self.filepath, 'w', newline='', buffering=1)
            self.writer = csv.writer(self.file)
            
            # Write comprehensive header
            self._write_header()
            
            self.iteration = 0
            
            print(f"📊 MPCC Unified Logger initialized: {self.filepath}")
            
        except Exception as e:
            print(f"Logger initialization failed: {e}")
            self.close()
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and termination signals"""
        print(f"\n Signal {signum} received, closing logger...")
        self.close()
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)
    
    def _write_header(self):
        """Write comprehensive CSV header"""
        header = [
            # Meta
            'iteration', 'timestamp',
            
            # State (current)
            'X', 'Y', 'phi_deg', 'v_state',
            
            # Progress
            'theta', 'theta_normalized', 'lap_progress_pct',
            
            # Control (applied)
            'delta_deg', 'delta_rad', 'acceleration',
            
            # Virtual velocity
            'v_virtual',
            
            # Errors
            'e_c', 'e_l', 'distance_to_centerline',
            
            # Costs
            'cost_total', 'cost_contouring', 'cost_lag', 
            'cost_progress', 'cost_input', 'cost_slack',
            
            # Cost weights (for reference)
            'q_c', 'q_l', 'gamma', 'R_u', 'q_slack',
            
            # Solver performance
            'solver_status', 'solver_iterations', 'solve_time_ms',
            'obj_value', 'primal_residual', 'dual_residual',
            
            # Horizon info
            'horizon_steps', 'dt',
            
            # Constraint violations (aggregated)
            'delta_violation', 'accel_violation',
            'v_state_violation', 'v_virtual_violation',
            'theta_violation', 'track_violation',
            'slack_violation',
            
            # Predicted trajectory summary (optional - first 3 steps)
            'x_pred_1_X', 'x_pred_1_Y', 'x_pred_1_v',
            'x_pred_2_X', 'x_pred_2_Y', 'x_pred_2_v',
            'x_pred_3_X', 'x_pred_3_Y', 'x_pred_3_v',
            
            # Reference tracking
            'X_ref', 'Y_ref', 'Phi_ref_deg',
        ]
        
        self.writer.writerow(header)
    
    def log_iteration(self, timestamp, state, theta, u_opt, x_pred, theta_pred, 
                     solver_result, track, controller, v_virtual_pred=None):
        """
        Log complete MPCC iteration to single CSV row
        
        All exceptions caught - won't crash controller!
        """
        if not self.enable or self.closed:
            return
        
        try:
            self.iteration += 1
            
            # ================================================================
            # Extract current state
            # ================================================================
            X, Y, phi, v = state
            delta_opt = float(u_opt[0])
            a_opt = float(u_opt[1])
            
            # ================================================================
            # Compute errors
            # ================================================================
            e_c, e_l = track.compute_errors(X, Y, theta)
            dist_to_centerline = abs(e_c)
            lap_progress = (theta / track.L) * 100.0
            
            # Get reference
            X_ref, Y_ref, Phi = track.get_reference(theta)
            
            # ================================================================
            # Virtual velocity
            # ================================================================
            v_virtual = v_virtual_pred[0] if (v_virtual_pred is not None and len(v_virtual_pred) > 0) else 0.0
            
            # ================================================================
            # Compute costs
            # ================================================================
            costs = self._compute_costs(state, theta, u_opt, track, controller)
            
            # ================================================================
            # Solver stats
            # ================================================================
            if hasattr(solver_result, 'info'):
                info = solver_result.info
                solver_status = getattr(info, 'status', 'unknown')
                solver_iters = getattr(info, 'iter', 0)
                solve_time = getattr(info, 'solve_time', 0) * 1000  # ms
                obj_val = getattr(info, 'obj_val', 0)
                pri_res = getattr(info, 'pri_res', 0)
                dua_res = getattr(info, 'dua_res', 0)
            else:
                solver_status = 'no_result'
                solver_iters = 0
                solve_time = 0
                obj_val = 0
                pri_res = 0
                dua_res = 0
            
            # ================================================================
            # Constraint violations (aggregated)
            # ================================================================
            delta_viol = max(0, abs(delta_opt) - controller.delta_max)
            accel_viol = max(0, abs(a_opt) - controller.a_max)
            v_state_viol = max(0, -v) + max(0, v - controller.v_max)
            v_virtual_viol = max(0, -v_virtual) + max(0, v_virtual - controller.v_max)
            theta_viol = max(0, -theta) + max(0, theta - track.L)
            
            # Track boundary violation
            F, f = track.get_halfspace_constraints(theta)
            track_left_val = F[0, 0] * X + F[0, 1] * Y
            track_right_val = F[1, 0] * X + F[1, 1] * Y
            track_viol = max(0, track_left_val - f[0]) + max(0, track_right_val - f[1])
            
            slack_viol = 0.0  # Would need from solver
            
            # ================================================================
            # Predicted trajectory summary (first 3 steps)
            # ================================================================
            x_pred_1 = x_pred[0] if len(x_pred) > 0 else [0, 0, 0, 0]
            x_pred_2 = x_pred[1] if len(x_pred) > 1 else [0, 0, 0, 0]
            x_pred_3 = x_pred[2] if len(x_pred) > 2 else [0, 0, 0, 0]
            
            # ================================================================
            # Write unified row
            # ================================================================
            row = [
                # Meta
                self.iteration, timestamp,
                
                # State
                X, Y, np.rad2deg(phi), v,
                
                # Progress
                theta, np.mod(theta, track.L), lap_progress,
                
                # Control
                np.rad2deg(delta_opt), delta_opt, a_opt,
                
                # Virtual velocity
                v_virtual,
                
                # Errors
                e_c, e_l, dist_to_centerline,
                
                # Costs
                costs['total'], costs['contouring'], costs['lag'],
                costs['progress'], costs['input'], costs['slack'],
                
                # Cost weights
                controller.q_c, controller.q_l, controller.gamma,
                controller.R_u, controller.q_slack,
                
                # Solver
                solver_status, solver_iters, solve_time,
                obj_val, pri_res, dua_res,
                
                # Horizon
                len(x_pred), controller.dt,
                
                # Violations
                delta_viol, accel_viol, v_state_viol,
                v_virtual_viol, theta_viol, track_viol, slack_viol,
                
                # Predictions
                x_pred_1[0], x_pred_1[1], x_pred_1[3],
                x_pred_2[0], x_pred_2[1], x_pred_2[3],
                x_pred_3[0], x_pred_3[1], x_pred_3[3],
                
                # Reference
                X_ref, Y_ref, np.rad2deg(Phi),
            ]
            
            self.writer.writerow(row)
            # Line buffering ensures immediate write
            
        except Exception as e:
            print(f" Logger error (iteration {self.iteration}): {e}")
    
    def _compute_costs(self, state, theta, u, track, controller):
        """Compute individual cost components"""
        try:
            X, Y, phi, v = state
            delta, a = u
            
            X_ref, Y_ref, Phi = track.get_reference(theta)
            
            e_c = np.sin(Phi) * (X - X_ref) - np.cos(Phi) * (Y - Y_ref)
            e_l = -np.cos(Phi) * (X - X_ref) - np.sin(Phi) * (Y - Y_ref)
            
            cost_c = controller.q_c * e_c**2
            cost_l = controller.q_l * e_l**2
            cost_p = 0.0  # Progress cost (would need v_virtual)
            cost_u = controller.R_u * (delta**2 + a**2)
            cost_s = 0.0  # Slack cost (would need from solver)
            
            cost_total = cost_c + cost_l + cost_p + cost_u + cost_s
            
            return {
                'total': cost_total,
                'contouring': cost_c,
                'lag': cost_l,
                'progress': cost_p,
                'input': cost_u,
                'slack': cost_s
            }
        except Exception:
            return {'total': 0, 'contouring': 0, 'lag': 0, 'progress': 0, 'input': 0, 'slack': 0}
    
    def close(self):
        """Close log file"""
        if not self.enable or self.closed:
            return
        
        self.closed = True
        
        try:
            if hasattr(self, 'file') and self.file and not self.file.closed:
                self.file.flush()
                self.file.close()
        except Exception as e:
            print(f" Error closing log file: {e}")
        
        if hasattr(self, 'filepath'):
            print(f"MPCC Unified Logger closed: {self.filepath}")
            print(f"Total iterations logged: {self.iteration}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
    
    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except:
            pass