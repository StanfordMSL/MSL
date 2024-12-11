import time
import shutil
import os
import numpy as np
import scipy.linalg
import splines.min_snap as ms
import utilities.trajectory_helper as th
import utilities.dynamics_helper as dh

from controller.base_controller import BaseController
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import vertcat
from dynamics.quadcopter_model import export_quadcopter_ode_model
from typing import Union,Tuple,Dict
from copy import deepcopy
# import visualize.plot_synthesize as ps

class VehicleRateMPC(BaseController):
    def __init__(self,
                 fout_wps:Dict[str,Dict[str,Union[float,np.ndarray]]],
                 mpc_prms:Dict[str,Union[float,np.ndarray]],
                 drn_prms:Dict[str,Union[float,np.ndarray]],
                 ctl_prms:Dict[str,Union[float,np.ndarray]],
                 name:str="vrmpc") -> None:
        
        """
        Constructor for the VehicleRateMPC class.
        
        Args:
            - fout_wps: Dictionary containing the flat output waypoints.
            - mpc_prms: Dictionary containing the MPC parameters.
            - drn_prms: Dictionary containing the drone parameters.
            - ctl_prms: Dictionary containing the controller parameters.
            - hz_ctrl:  Controller frequency.
            - name:     Name of the policy.

        Variables:
            - name:    Name of the policy.
            - Nx:      Number of states.
            - Nu:      Number of inputs.
            - tXUd:    Ideal trajectory.
            - Qk:      State cost matrix.
            - Rk:      Input cost matrix.
            - QN:      Terminal state cost matrix.
            - lbu:     Lower bound on inputs.
            - ubu:     Upper bound on inputs.
            - wts:     Search weights for xv_ds.
            - ns:      Search window size for xv_ds.
            - hz:      Frequency of the MPC rollout.
            - use_RTI: Use RTI flag.
            - model:   Acados OCP model.
            - solver:  Acados OCP Solver.

            - export_dir:       Directory where the code is exported.
            - solver_path:      Path to the solver json file.
            - simulator_path:   Path to the simulator json file.

        """

        # =====================================================================
        # Extract parameters
        # =====================================================================

        # MPC Parameters
        Nhn = mpc_prms["horizon"]
        Qk,Rk,QN = np.diag(mpc_prms["Qk"]),np.diag(mpc_prms["Rk"]),np.diag(mpc_prms["QN"])
        Ws = np.diag(mpc_prms["Ws"])

        # Control Parameters
        hz_ctl,use_RTI,tpad= ctl_prms["hz"],ctl_prms["use_RTI"],ctl_prms["terminal_padding"]

        # Derived Parameters
        traj_config_pd = self.pad_trajectory(fout_wps,Nhn,hz_ctl,tpad)
        drn_spec = dh.generate_specifications(drn_prms,ctl_prms)
        nx,nu = drn_spec["nx_br"], drn_spec["nu_br"]
        lbu,ubu = drn_spec["lbu"],drn_spec["lbu"]

        ny,ny_e = nx+nu,nx
        solver_json = 'acados_ocp_nlp_'+name+'.json'
        
        # =====================================================================
        # Compute Desired Trajectory
        # =====================================================================

        # Solve Padded Trajectory
        output = ms.solve(traj_config_pd)
        if output is not False:
            Tpi, CPi = output
        else:
            raise ValueError("Padded trajectory (for VehicleRateMPC) not feasible. Aborting.")
        
        # Convert to desired tXU
        tXUd = th.ts_to_tXU(Tpi,CPi,drn_spec,hz_ctl)
        
        # =====================================================================
        # Setup Acados Variables
        # =====================================================================

        # Initialize Acados OCP
        ocp = AcadosOcp()
        ocp.dims.N = Nhn

        ocp.model = export_quadcopter_ode_model(drn_spec["m"],drn_spec["tn"])        
        ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Qk,Rk)
        ocp.cost.W_e = QN

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        ocp.constraints.x0 = tXUd[1:11,0]
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Initialize Acados Solver
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10

        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        ocp.solver_options.qp_solver_cond_N = Nhn
        ocp.solver_options.tf = Nhn/hz_ctl
        ocp.solver_options.qp_solver_warm_start = 1
        
        # =====================================================================
        # Controller Variables
        # =====================================================================

        self.name = "VehicleRateMPC"
        self.Nx,self.Nu = nx,nu
        self.tXUd = tXUd
        self.Qk,self.Rk,self.QN = Qk,Rk,QN
        self.Ws = Ws
        self.lbu,self.ubu = lbu,ubu
        self.ns = int(hz_ctl/5)
        self.hz = hz_ctl
        self.use_RTI = use_RTI
        self.model = ocp.model
        self.solver = AcadosOcpSolver(ocp,json_file=solver_json,verbose=False)
        self.export_dir = ocp.code_export_directory
        self.solver_path = os.path.join(os.path.dirname(self.export_dir),solver_json)
        self.simulator_path = None

        # =====================================================================
        # Warm start the solver
        # =====================================================================
        
        for _ in range(5):
            self.control(0.0,tXUd[1:11,0])

    def control(self,
                tcr:float,xcr:np.ndarray,
                upr:np.ndarray=None,
                obj:np.ndarray=None,
                icr:None=None,zcr:None=None) -> Tuple[
                    np.ndarray,None,None,np.ndarray]:
        
        """
        Method to compute the control input for the VehicleRateMPC controller. We use the standard input arguments
        format with the unused arguments set to None. Likewise, we use the standard output format with the unused
        outputs set to None.

        Args:
        tcr:    Current time.
        xcr:    Current state.
        upr:    Previous control input (unused).
        obj:    Objective (unused).
        icr:    Current image (unused).
        zcr:    Current image feature vector (unused).

        Returns:
        ucc:    Control input.
        None:   Output feature vector (unused).
        None:   Oracle output  (unused).
        tsol:   Time taken to solve components [setup ocp, solve ocp, unused, unused].

        """
        # Unused arguments
        _ = upr,obj,icr,zcr

        # Start timer
        t0 = time.time()

        # Get desired trajectory
        ydes = self.get_ydes(xcr,tcr)

        # Set desired trajectory
        for i in range(self.solver.acados_ocp.dims.N):
            self.solver.cost_set(i, "yref", ydes[:,i])
        self.solver.cost_set(self.solver.acados_ocp.dims.N, "yref", ydes[0:10,-1])
        
        # Solve OCP
        t1 = time.time()
        if self.use_RTI:
            # preparation phase
            self.solver.options_set('rti_phase', 1)
            status = self.solver.solve()

            # set initial state
            self.solver.set(0, "lbx", xcr)
            self.solver.set(0, "ubx", xcr)

            # feedback phase
            self.solver.options_set('rti_phase', 2)
            status = self.solver.solve()

            ucc = self.solver.get(0, "u")
        else:
            # Solve ocp and get next control input
            try:
                ucc = self.solver.solve_for_x0(x0_bar=xcr)
            except:
                print("Warning: VehicleRateMPC failed to solve OCP. Using previous input.")
                ucc = self.solver.get(0, "u")
        t2 = time.time()

        # Compute timer values
        tsol = np.array([t1-t0,t2-t1,0.0,0.0])

        return ucc,None,None,tsol

    def pad_trajectory(self,fout_wps:Dict[str,Union[str,int,Dict[str,Union[float,np.ndarray]]]],
                       Nhn:int,hz_ctl:float,tpad:float) -> Dict[str,Dict[str,Union[float,np.ndarray]]]:
        """
        Method to pad the trajectory with the final waypoint so that the MPC horizon is satisfied at the end of the trajectory.

        Args:
        fout_wps:   Dictionary containing the flat output waypoints.
        Nhn:        Prediction horizon.
        hz_ctl:     Controller frequency.
        tpad:       Padding time.

        Returns:
        fout_wps_pd: Padded flat output waypoints.

        """

        # Get final waypoint
        kff = list(fout_wps["keyframes"])[-1]
        
        # Pad trajectory
        t_pd = fout_wps["keyframes"][kff]["t"]+Nhn/hz_ctl+tpad
        fo_pd = np.array(fout_wps["keyframes"][kff]["fo"])[:,0:3].tolist()

        fout_wps_pd = deepcopy(fout_wps)
        fout_wps_pd["keyframes"]["fof"] = {
            "t":t_pd,
            "fo":fo_pd}

        return fout_wps_pd

    def get_ydes(self,xcr:np.ndarray,ti:float) -> np.ndarray:
        """
        Method to get the section of the desired trajectory at the current time.

        Args:
        xcr:    Current state.
        ti:     Current time.

        Returns:
        ydes:   Desired trajectory section at the current time.

        """
        # Get relevant portion of trajectory
        idx_i = int(self.hz*ti)
        ks0 = np.clip(idx_i-self.ns,0,self.tXUd.shape[1]-1)
        ksf = np.min([idx_i+self.ns,self.tXUd.shape[1]])
        xi = self.tXUd[1:11,ks0:ksf]

        # Find index of nearest state
        dx = xi-xcr.reshape(-1,1)
        idx0 = ks0 + np.argmin(self.Ws.T@dx**2)
        idxf = idx0 + self.solver.acados_ocp.dims.N+1

        # Pad if idxf is greater than the last index
        if idxf < self.tXUd.shape[1]:
            xdes = self.tXUd[1:11,idx0:idxf]
            udes = self.tXUd[11:15,idx0:idxf]
            
            ydes = np.vstack((xdes,udes))
        else:
            print("Warning: VehicleRateMPC.get_ydes() padding trajectory. Increase your padding horizon.")
            xdes = self.tXUd[1:11,idx0:]
            udes = self.tXUd[11:15,idx0:]

            ydes = np.vstack((xdes,udes))
            ydes = np.hstack((ydes,np.tile(ydes[:,-1:],(1,idxf-self.tXUd.shape[1]))))

        return ydes

    def generate_simulator(self,hz):
        """
        Method to generate the Acados simulator for the VehicleRateMPC controller.

        Args:
        hz:    Frequency of the simulator.

        Returns:
        sim:   AcadosSimSolver object.
        """

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1/hz
        sim.solver_options.integrator_type = 'IRK'

        sim_json = 'acados_sim_nlp.json'
        self.simulator_path = os.path.join(os.path.dirname(self.code_export_directory),sim_json)

        return AcadosSimSolver(sim,json_file=sim_json,verbose=False)
    
    def clear_generated_code(self):
        """
        Method to clear the generated code and files to ensure the code is recompiled correctly each time.
        """
        
        try:
            os.remove(self.solver_path)
            shutil.rmtree(self.export_dir)
            os.remove(self.simulator_path)
        except:
            pass