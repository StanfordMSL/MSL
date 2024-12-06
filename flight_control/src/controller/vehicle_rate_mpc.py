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
                 name:str="policy") -> None:
        
        """
        Constructor for the VehicleRateMPC class.
        
        Args:
        fout_wps:   Dictionary containing the flat output waypoints.
        mpc_prms:   Dictionary containing the MPC parameters.
        drn_prms:   Dictionary containing the drone parameters.
        ctl_prms:   Dictionary containing the controller parameters.
        hz_ctrl:    Controller frequency.
        name:       Name of the policy.
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
        
        # Setup OCP variables
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

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10

        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        ocp.solver_options.qp_solver_cond_N = Nhn
        ocp.solver_options.tf = Nhn/hz_ctl
        ocp.solver_options.qp_solver_warm_start = 1
        
        # Controller Variables
        self.name = "VehicleRateMPC"
        self.Nx,self.Nu = nx,nu
        self.tXUd = tXUd                                                                        # Ideal Trajectory
        self.Qk,self.Rk,self.QN = Qk,Rk,QN                                                      # Cost Matrices
        self.lbu,self.ubu = lbu,ubu                                                             # Input Limits
        self.wts = Ws                                                              # Search weights for xv_ds
        self.ns = int(hz_ctl/5)                                                            # Search window size for xv_ds
        self.hz = hz_ctl                                                                   # Frequency of the MPC rollout
        self.use_RTI = use_RTI                                                                  # Use RTI flag
        self.model = ocp.model                                                                  # Acados OCP
        self.ocp_solver = AcadosOcpSolver(ocp,json_file=solver_json,verbose=False)            # Acados OCP Solver
        
        self.code_export_directory = ocp.code_export_directory
        self.solver_json_file = os.path.join(os.path.dirname(self.code_export_directory),solver_json)

        # do some initial iterations to start with a good initial guess
        for _ in range(5):
            self.control(np.zeros(4),0.0,tXUd[1:11,0],np.zeros(10))

    def control(self,
                upr:np.ndarray,
                tcr:float,xcr:np.ndarray,
                obj:np.ndarray,
                icr:None=None,zcr:None=None) -> Tuple[
                    np.ndarray,None,None,np.ndarray]:
        
        # Unused arguments
        _ = upr,obj,icr,zcr
        
        # Start timer
        t0 = time.time()

        # Get reference trajectory
        yref = self.get_yref(xcr,tcr)

        # Set reference trajectory
        for i in range(self.ocp_solver.acados_ocp.dims.N):
            self.ocp_solver.cost_set(i, "yref", yref[:,i])
        self.ocp_solver.cost_set(self.ocp_solver.acados_ocp.dims.N, "yref", yref[0:10,-1])
        
        t1 = time.time()
        if self.use_RTI:
            # preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            status = self.ocp_solver.solve()

            # set initial state
            self.ocp_solver.set(0, "lbx", xcr)
            self.ocp_solver.set(0, "ubx", xcr)

            # feedback phase
            self.ocp_solver.options_set('rti_phase', 2)
            status = self.ocp_solver.solve()

            ucc = self.ocp_solver.get(0, "u")
        else:
            # Solve ocp and get next control input
            try:
                ucc = self.ocp_solver.solve_for_x0(x0_bar=xcr)
            except:
                print("Warning: VehicleRateMPC failed to solve OCP. Using previous input.")
                ucc = self.ocp_solver.get(0, "u")

        t2 = time.time()

        # End timer
        tsol = np.array([t1-t0,t2-t1,0.0,0.0])

        return ucc,None,None,tsol

    def pad_trajectory(self,fout_wps:Dict[str,Union[str,int,Dict[str,Union[float,np.ndarray]]]],
                       Nhn:int,hz_ctl:float,tpad:float) -> Dict[str,Dict[str,Union[float,np.ndarray]]]:
        
        kff = list(fout_wps["keyframes"])[-1]
        t_pd = fout_wps["keyframes"][kff]["t"]+Nhn/hz_ctl+tpad
        fo_pd = np.array(fout_wps["keyframes"][kff]["fo"])[:,0:3].tolist()

        traj_config_pd = deepcopy(fout_wps)
        traj_config_pd["keyframes"]["fof"] = {
            "t":t_pd,
            "fo":fo_pd}

        return traj_config_pd

    def get_yref(self,xcr:np.ndarray,ti:float) -> np.ndarray:
        # Get relevant portion of trajectory
        idx_i = int(self.hz*ti)
        ks0 = np.clip(idx_i-self.ns,0,self.tXUd.shape[1]-1)
        ksf = np.min([idx_i+self.ns,self.tXUd.shape[1]])
        xi = self.tXUd[1:11,ks0:ksf]

        # Find index of nearest state
        dx = xi-xcr.reshape(-1,1)
        idx0 = ks0 + np.argmin(self.wts.T@dx**2)
        idxf = idx0 + self.ocp_solver.acados_ocp.dims.N+1

        # Pad if idxf is greater than the last index
        if idxf < self.tXUd.shape[1]:
            xref = self.tXUd[1:11,idx0:idxf]
            uref = self.tXUd[11:15,idx0:idxf]
            
            yref = np.vstack((xref,uref))
        else:
            print("Warning: VehicleRateMPC.get_yref() padding trajectory. Increase your padding horizon.")
            xref = self.tXUd[1:11,idx0:]
            uref = self.tXUd[11:15,idx0:]

            yref = np.vstack((xref,uref))
            yref = np.hstack((yref,np.tile(yref[:,-1:],(1,idxf-self.tXUd.shape[1]))))

        return yref

    def generate_simulator(self,hz):
        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1/hz
        sim.solver_options.integrator_type = 'IRK'

        sim_json = 'acados_sim_nlp.json'
        self.sim_json_file = os.path.join(os.path.dirname(self.code_export_directory),sim_json)

        return AcadosSimSolver(sim,json_file=sim_json,verbose=False)
    
    def clear_generated_code(self):
        try:
            os.remove(self.solver_json_file)
            shutil.rmtree(self.code_export_directory)
            os.remove(self.sim_json_file)
        except:
            pass