
"""
ROS2 node for MPCC controller

Subscribes to:
    /odom (nav_msgs/Odometry): Vehicle odometry
    
Publishes to:
    /drive (ackermann_msgs/AckermannDriveStamped): Control commands
    /mpcc/predicted_path (nav_msgs/Path): Predicted trajectory
    /mpcc/reference_path (nav_msgs/Path): Reference centerline
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import csv
import pathlib 
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA

from mpcc_controller.track_map import TrackMap
from mpcc_controller.vehicle_model import VehicleModel
from mpcc_controller.mpcc_controller import MPCCController
from mpcc_controller.mpcc_logger import MPCCLogger


class MPCCNode(Node):
    """ROS2 node for MPCC control"""
    
    def __init__(self):
        super().__init__('mpcc_controller')
        
        # Declare parameters
        self.declare_parameter('waypoints_file', 'waypoints.csv')
        self.declare_parameter('horizon_length', 10)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('track_width', 1.0)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('visualize', True)
        
        # Cost weights
        self.declare_parameter('q_contour', 10.0)
        self.declare_parameter('q_lag', 100.0)
        self.declare_parameter('gamma', 50.0)
        
        # Vehicle constraints
        self.declare_parameter('max_steering', 0.4)
        self.declare_parameter('max_acceleration', 3.0)
        self.declare_parameter('max_velocity', 8.0)
        
        # Get parameters
        waypoints_file = self.get_parameter('waypoints_file').value
        N = self.get_parameter('horizon_length').value
        dt = self.get_parameter('dt').value
        wheelbase = self.get_parameter('wheelbase').value
        track_width = self.get_parameter('track_width').value
        self.control_freq = self.get_parameter('control_frequency').value
        self.visualize = self.get_parameter('visualize').value
        
        # Load waypoints
        self.get_logger().info(f'Loading waypoints from: {waypoints_file}')
        waypoints = self._load_waypoints(waypoints_file)
        self.get_logger().info(f'Loaded {len(waypoints)} waypoints')
        print(f"First waypoint: {waypoints[0]}")
        print(f"Waypoint range X: [{waypoints[:,0].min():.3f}, {waypoints[:,0].max():.3f}]")
        print(f"Waypoint range Y: [{waypoints[:,1].min():.3f}, {waypoints[:,1].max():.3f}]")
                
        # Initialize track and vehicle
        self.track = TrackMap(waypoints, track_width)
        print(f"🔍 TRACK LENGTH = {self.track.L:.3f} meters")  # ADD THIS!
        self.vehicle = VehicleModel(wheelbase)
        
        # Initialize MPCC controller
        self.controller = MPCCController(self.vehicle, self.track, N, dt)
        
        # Set cost weights from parameters
        self.controller.q_c = self.get_parameter('q_contour').value
        self.controller.q_l = self.get_parameter('q_lag').value
        self.controller.gamma = self.get_parameter('gamma').value
        
        # Set constraints
        self.controller.delta_max = self.get_parameter('max_steering').value
        self.controller.a_max = self.get_parameter('max_acceleration').value
        self.controller.v_max = self.get_parameter('max_velocity').value
        
        # State variables
        self.state = None  # [X, Y, φ, v]
        self.theta = None  # Current progress
        self.initialized = False
        
        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            PoseWithCovarianceStamped,  # Changed message type
            '/amcl_pose',               # Changed topic name!
            self.pose_callback,         # Renamed callback
            qos
        )
        
        # Publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            qos
        )
        
        if self.visualize:
            self.pred_path_pub = self.create_publisher(Path, '/mpcc/predicted_path', qos)
            self.ref_path_pub = self.create_publisher(Path, '/mpcc/reference_path', qos)
            self.markers_pub = self.create_publisher(MarkerArray, '/mpcc/diagnostics', qos)
        
        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)
                
        self.get_logger().info('MPCC Controller initialized')
        self.get_logger().info(f'Control frequency: {self.control_freq} Hz')
        self.get_logger().info(f'Horizon: {N} steps, dt: {dt}s')

        self.create_timer(0.1, self._check_shutdown)

        log_dir = pathlib.Path.home() / 'mpcc_logs'  # Or any directory you prefer
        self.logger = MPCCLogger(log_dir=str(log_dir), enable=True)
        self.get_logger().info(f'CSV Logger initialized: {log_dir}')

    def _check_shutdown(self):
        """Periodic check for shutdown - ensures logger cleanup"""
        # This will be called periodically, no action needed
        # The logger's atexit handler will clean up automatically
        pass
    
    def destroy_node(self):
        """Override destroy to ensure logger cleanup"""
        if hasattr(self, 'logger'):
            self.logger.close()
        super().destroy_node()
    
    def _load_waypoints(self, filepath):
        """Load waypoints from CSV file"""
        waypoints = []
        # Try to find file
        if not pathlib.Path(filepath).exists():
            # Try in package share directory
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('mpcc_controller')
            filepath = pathlib.Path(pkg_dir) / 'config' / filepath
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['x_m'])
                y = float(row['y_m'])
                waypoints.append([x, y])

        waypoints = list(reversed(waypoints))
        return np.array(waypoints)
    

    def pose_callback(self, msg):
        """Process AMCL pose message"""
        X = msg.pose.pose.position.x
        Y = msg.pose.pose.position.y
        
        # Extract heading from quaternion (same as before)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Convert to yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        phi = np.arctan2(siny_cosp, cosy_cosp)
        
        if self.state is not None:
            # Use previous velocity or estimate from position change
            dt = 0.05  # Control timestep
            if hasattr(self, 'last_position'):
                dx = X - self.last_position[0]
                dy = Y - self.last_position[1]
                v = np.sqrt(dx**2 + dy**2) / dt
            else:
                v = 0.0
        else:
            v = 0.0
        
        # Store for next iteration
        self.last_position = (X, Y)
        
        # Update state
        self.state = np.array([X, Y, phi, v])
        self.get_logger().info(
            f'State: X={self.state[0]:.2f}, Y={self.state[1]:.2f}, '
            f'φ={np.rad2deg(self.state[2]):.1f}°, v={self.state[3]:.2f}m/s',
            throttle_duration_sec=1.0
        )
        
        # Initialize theta on first message
        if not self.initialized:
            self.initialized = True
            self.theta = self.track.project_point(X, Y)
            self.get_logger().info(f'Initialized at θ={self.theta:.2f}m, pose=({X:.2f}, {Y:.2f})')

        
    
    def control_loop(self):
        """Main control loop (called at control_frequency)"""   
        if not self.initialized or self.state is None:
            return      
        
        try:
            timestamp = self.get_clock().now().nanoseconds / 1e9

            X, Y, phi, v = self.state
            X_ref, Y_ref, Phi_ref = self.track.get_reference(self.theta)
            e_c_now, e_l_now = self.track.compute_errors(X, Y, self.theta)


            print(f"\n{'='*70}")
            print(f"MPCC TICK  t={timestamp:.3f}s")
            print(f"{'='*70}")
            print(f"[INPUT TO SOLVER]")
            print(f"  State x0:")
            print(f"    X   = {X:+.4f} m")
            print(f"    Y   = {Y:+.4f} m")
            print(f"    phi = {np.rad2deg(phi):+.2f} deg")
            print(f"    v   = {v:+.4f} m/s")
            print(f"  Progress theta0 = {self.theta:.4f} m  (L = {self.track.L:.2f} m)")
            print(f"  Reference at theta0:")
            print(f"    X_ref   = {X_ref:+.4f} m")
            print(f"    Y_ref   = {Y_ref:+.4f} m")
            print(f"    Phi_ref = {np.rad2deg(Phi_ref):+.2f} deg")
            print(f"  Errors at current state:")
            print(f"    e_c (contouring) = {e_c_now:+.4f} m")
            print(f"    e_l (lag)        = {e_l_now:+.4f} m")
            print(f"    distance to ref  = {np.hypot(X-X_ref, Y-Y_ref):.4f} m")

            
            # Solve MPCC
            u_opt, x_pred, theta_pred = self.controller.solve(self.state, self.theta)

            v_virtual_pred = getattr(self.controller, 'v_virtual_prev', None)
        
            # Log iteration to CSV
            self.logger.log_iteration(
                timestamp=timestamp,
                state=self.state,
                theta=self.theta,
                u_opt=u_opt,
                x_pred=x_pred,
                theta_pred=theta_pred,
                solver_result=self.controller.last_result,
                track=self.track,
                controller=self.controller,
                v_virtual_pred=v_virtual_pred
            )
                
            # Extract control
            delta = float(u_opt[0])  # Steering
            a = float(u_opt[1])      # Acceleration

            print(f"[SOLVER OUTPUT]")
            if hasattr(self.controller, 'last_result') and self.controller.last_result is not None:
                info = self.controller.last_result.info
                print(f"  Status:      {info.status}")
                print(f"  Iterations:  {info.iter}")
                print(f"  Solve time:  {info.solve_time*1000:.2f} ms")
                print(f"  Objective:   {info.obj_val:.4f}")
            print(f"  Control u[0]:")
            print(f"    delta = {np.rad2deg(delta):+.2f} deg  "
                  f"({delta:+.4f} rad)")
            print(f"    a     = {a:+.3f} m/s^2")
            if v_virtual_pred is not None and len(v_virtual_pred) > 0:
                print(f"  Virtual velocity v_virt[0] = {v_virtual_pred[0]:+.4f} m/s")
                print(f"  Virtual velocity trajectory: "
                      f"min={v_virtual_pred.min():+.3f}, "
                      f"max={v_virtual_pred.max():+.3f}, "
                      f"mean={v_virtual_pred.mean():+.3f} m/s")

            # ============================================================
            # PREDICTED TRAJECTORY (first 3 steps of horizon)
            # ============================================================
            print(f"[PREDICTED TRAJECTORY]")
            n_show = min(3, len(x_pred))
            for k in range(n_show):
                print(f"  Step {k+1}: "
                      f"X={x_pred[k,0]:+.4f}, "
                      f"Y={x_pred[k,1]:+.4f}, "
                      f"phi={np.rad2deg(x_pred[k,2]):+.2f}deg, "
                      f"v={x_pred[k,3]:+.4f}m/s, "
                      f"theta={theta_pred[k]:.4f}m")
            print(f"  Horizon progress: "
                  f"delta_theta = {theta_pred[-1]-self.theta:+.4f} m "
                  f"over {len(theta_pred)} steps")
            print(f"{'='*70}\n")
            
            # Convert acceleration to target velocity
            v_current = self.state[3]
            v_target = v_current + a * self.controller.dt
            v_target = np.clip(v_target, 0.0, self.controller.v_max)
            
            # Clip steering
            delta = np.clip(delta, -self.controller.delta_max, self.controller.delta_max)
            
            # Publish drive command
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.steering_angle = delta
            drive_msg.drive.speed = v_target
            self.drive_pub.publish(drive_msg)
            
            # Update theta for next iteration (use prediction)
            # In pose_callback(), CHANGE line 243 to:
            theta_predicted = float(theta_pred[0])

            # Secondary source: projection of current pose, used only as
            # a drift correction with a small gain.
            theta_measured = self.track.project_point(self.state[0], self.state[1])

            # Wrap-aware error: shortest signed distance on the circular
            # track, in the range (-L/2, +L/2].
            L = self.track.L
            theta_error = (theta_measured - theta_predicted + L / 2.0) % L - L / 2.0

            correction_gain = 0.05
            self.theta = (theta_predicted + correction_gain * theta_error) % L
            
            # Compute errors for logging
            e_c, e_l = self.track.compute_errors(
                self.state[0], self.state[1], self.theta
            )
            
            # Log
            self.get_logger().info(
                f'θ={self.theta:.2f}m, v={v_current:.2f}m/s, '
                f'δ={np.rad2deg(delta):.1f}°, a={a:.2f}m/s², '
                f'e_c={e_c:.3f}m, e_l={e_l:.3f}m',
                throttle_duration_sec=1.0
            )
            
            # Visualization
            if self.visualize:
                self._publish_visualization(x_pred, theta_pred)
                
        except Exception as e:
            import traceback
            self.get_logger().error(f'Control loop error: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')
    
    def _publish_visualization(self, x_pred, theta_pred):
        """Publish visualization markers"""
        stamp = self.get_clock().now().to_msg()
        
        # Predicted path
        pred_path = Path()
        pred_path.header.stamp = stamp
        pred_path.header.frame_id = 'map'
        
        for x in x_pred:
            pose = PoseStamped()
            pose.header = pred_path.header
            pose.pose.position.x = x[0]
            pose.pose.position.y = x[1]
            pred_path.poses.append(pose)
        
        self.pred_path_pub.publish(pred_path)
        
        # Reference path
        ref_path = Path()
        ref_path.header = pred_path.header
        
        for theta in theta_pred:
            X_ref, Y_ref, _ = self.track.get_reference(theta)
            pose = PoseStamped()
            pose.header = ref_path.header
            pose.pose.position.x = X_ref
            pose.pose.position.y = Y_ref
            ref_path.poses.append(pose)
        
        self.ref_path_pub.publish(ref_path)

    def __del__(self):
        """Destructor - close logger on shutdown"""
        if hasattr(self, 'logger'):
            self.logger.close()


def main(args=None):
    rclpy.init(args=args)
    
    node = MPCCNode()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("⚠️ Keyboard interrupt")
    except Exception as e:
        print(f"❌ Executor error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 Cleaning up...")
        if hasattr(node, 'logger'):
            node.logger.close()
        print("🛑 Shutting down...")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("✅ Shutdown complete")



if __name__ == '__main__':
    main()
