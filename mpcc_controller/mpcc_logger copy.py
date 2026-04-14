#!/usr/bin/env python3
"""
DEBUGGING VERSION: Verbose logging to find why files are empty
"""

import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import atexit
import signal


class MPCCLogger:
    """Logger with debug prints"""
    
    def __init__(self, log_dir='mpcc_logs', enable=True):
        self.enable = enable
        self.closed = False
        self.files = {}
        
        if not self.enable:
            return
        
        # Create directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / f'run_{timestamp}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG: Created directory: {self.log_dir}")
        
        # Register cleanup
        atexit.register(self.close)
        
        try:
            self._init_main_log()
            self._init_states_log()
            self._init_controls_log()
            self._init_costs_log()
            self._init_constraints_log()
            self._init_solver_log()
            
            self.iteration = 0
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.close()
            raise
    
    def _open_csv_file(self, filename, header):
        """Open CSV with debug output"""
        filepath = self.log_dir / filename
        
        file_handle = open(filepath, 'w', newline='', buffering=1)
        writer = csv.writer(file_handle)
        writer.writerow(header)
        
        self.files[filename] = file_handle
        
        return file_handle, writer
    
    def _init_main_log(self):
        header = [
            'iteration', 'timestamp',
            'X', 'Y', 'phi_deg', 'v',
            'theta', 'theta_normalized',
            'delta_deg', 'acceleration',
            'v_virtual',
            'cost_total', 'cost_contouring', 'cost_lag', 
            'cost_progress', 'cost_input', 'cost_slack',
            'e_c', 'e_l',
            'distance_to_centerline', 'lap_progress_pct'
        ]
        self.main_file, self.main_writer = self._open_csv_file('main_log.csv', header)
    
    def _init_states_log(self):
        header = [
            'iteration', 'timestamp', 'step',
            'X', 'Y', 'phi_deg', 'v',
            'theta', 'v_virtual',
            'step_type'
        ]
        self.states_file, self.states_writer = self._open_csv_file('states_log.csv', header)
    
    def _init_controls_log(self):
        header = [
            'iteration', 'timestamp', 'step',
            'delta_deg', 'delta_rad', 'acceleration',
            'step_type'
        ]
        self.controls_file, self.controls_writer = self._open_csv_file('controls_log.csv', header)
    
    def _init_costs_log(self):
        header = [
            'iteration', 'timestamp', 'step',
            'cost_contouring', 'cost_lag', 'cost_progress',
            'cost_input', 'cost_slack',
            'e_c', 'e_l',
            'q_c', 'q_l', 'gamma', 'R_u', 'q_slack'
        ]
        self.costs_file, self.costs_writer = self._open_csv_file('costs_log.csv', header)
    
    def _init_constraints_log(self):
        header = [
            'iteration', 'timestamp', 'step',
            'delta_used', 'delta_max', 'delta_violation',
            'accel_used', 'accel_max', 'accel_violation',
            'v_state', 'v_state_min', 'v_state_max', 'v_state_violation',
            'v_virtual', 'v_virtual_min', 'v_virtual_max', 'v_virtual_violation',
            'v_coupling_slack', 'v_coupling_violation',
            'theta', 'theta_min', 'theta_max', 'theta_violation',
            'slack_left', 'slack_right',
            'track_left_margin', 'track_right_margin',
            'track_violation'
        ]
        self.constraints_file, self.constraints_writer = self._open_csv_file('constraints_log.csv', header)
    
    def _init_solver_log(self):
        header = [
            'iteration', 'timestamp',
            'status', 'iterations', 'solve_time_ms',
            'obj_value', 'primal_residual', 'dual_residual',
            'horizon_steps', 'dt'
        ]
        self.solver_file, self.solver_writer = self._open_csv_file('solver_log.csv', header)
    
    def log_iteration(self, timestamp, state, theta, u_opt, x_pred, theta_pred, 
                     solver_result, track, controller, v_virtual_pred=None):
        """Log with verbose debug output"""
        
        if not self.enable:
            return
            
        if self.closed:
            return
        
        try:
            self.iteration += 1
            
            # Extract values
            X, Y, phi, v = state
            delta_opt = float(u_opt[0]) 
            a_opt = float(u_opt[1])
            
            # Compute errors
            e_c, e_l = track.compute_errors(X, Y, theta)
            dist_to_centerline = abs(e_c)
            lap_progress = (theta / track.L) * 100.0
            
            # Compute costs
            costs = self._compute_costs(state, theta, u_opt, track, controller)
            
            # Virtual velocity
            v_virtual = v_virtual_pred[0] if (v_virtual_pred is not None and len(v_virtual_pred) > 0) else 0.0
            
            # === MAIN LOG ===
            row = [
                self.iteration, timestamp,
                X, Y, np.rad2deg(phi), v,
                theta, np.mod(theta, track.L),
                np.rad2deg(delta_opt), a_opt,
                v_virtual,
                costs['total'], costs['contouring'], costs['lag'],
                costs['progress'], costs['input'], costs['slack'],
                e_c, e_l,
                dist_to_centerline, lap_progress
            ]
            self.main_writer.writerow(row)
            self.main_file.flush()  # Force flush for debugging
            
            # Minimal logging for other files (just to test)
            self.states_writer.writerow([
                self.iteration, timestamp, 0,
                X, Y, np.rad2deg(phi), v,
                theta, v_virtual, 'current'
            ])
            self.states_file.flush()
        
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def _compute_costs(self, state, theta, u, track, controller):
        """Compute costs (simplified for debugging)"""
        try:
            X, Y, phi, v = state
            delta, a = u
            
            X_ref, Y_ref, Phi = track.get_reference(theta)
            
            e_c = np.sin(Phi) * (X - X_ref) - np.cos(Phi) * (Y - Y_ref)
            e_l = -np.cos(Phi) * (X - X_ref) - np.sin(Phi) * (Y - Y_ref)
            
            cost_c = controller.q_c * e_c**2
            cost_l = controller.q_l * e_l**2
            cost_u = controller.R_u * (delta**2 + a**2)
            cost_total = cost_c + cost_l + cost_u
            
            return {
                'total': cost_total,
                'contouring': cost_c,
                'lag': cost_l,
                'progress': 0.0,
                'input': cost_u,
                'slack': 0.0
            }
        except Exception as e:
            return {'total': 0, 'contouring': 0, 'lag': 0, 'progress': 0, 'input': 0, 'slack': 0}
    
    def close(self):
        """Close with debug output"""
        if not self.enable:
            return
            
        if self.closed:
            return
        
        self.closed = True
        
        for filename, file_handle in self.files.items():
            try:
                if file_handle and not file_handle.closed:
                    file_handle.flush()
                    file_handle.close()
                    print(f" DEBUG: Closed {filename}")
            except Exception as e:
                print(f" Error closing {filename}: {e}")
        
        self.files.clear()
        
        if hasattr(self, 'log_dir'):
            print(f" MPCC Logger closed: {self.log_dir}")
            print(f" Total iterations logged: {self.iteration}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        try:
            self.close()
        except:
            pass