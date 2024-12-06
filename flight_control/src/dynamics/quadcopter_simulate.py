from re import T, X
import numpy as np
import torch
import os
import json
from typing import List,Union,Literal

import visualize.plot_synthesize as ps

from controller.pilot import Pilot
from controller.vehicle_rate_mpc import VehicleRateMPC
from synthesize.solvers import min_snap as ms
import synthesize.nerf_utils as nf
import dynamics.quadcopter_config as qc
import synthesize.trajectory_helper as th
import synthesize.generate_data as gd
import visualize.record_flight as rf
from torchvision.io import write_video
from torchvision.transforms import Resize
from acados_template import AcadosSimSolver

def simulate_roster(cohort:str,course_name:str,drone_name:str,method:str,roster:List[str],
                    nerf:nf.NeRF,Nsps:int=10,use_fr:bool=False):

    # Print Some Information
    print("=====================================================================")
    print("Course Name: ",course_name)
    print("Testing Method: ",method)

    # Generate some useful paths
    workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    course_path    = os.path.join(workspace_path,"configs","courses",course_name+".json")
    drone_path     = os.path.join(workspace_path,"configs","drones",drone_name+".json")
    method_path    = os.path.join(workspace_path,"configs","methods",method+".json")

    # Load Configs
    with open(course_path) as json_file:
        course_config = json.load(json_file)
    with open(drone_path) as json_file:
        drone_config = json.load(json_file)
    with open(method_path) as json_file:
        method_config = json.load(json_file)
        sample_set_config = method_config["sample_set"]
        trajectory_set_config = method_config["trajectory_set"]
        drone_set_config = method_config["drone_set"]

    # Unpack sample set config
    mu_md = np.array(sample_set_config["model_noise"]["mean"])
    std_md = np.array(sample_set_config["model_noise"]["std"])
    mu_sn = np.array(sample_set_config["sensor_noise"]["mean"])
    std_sn = np.array(sample_set_config["sensor_noise"]["std"])
    outdoors = sample_set_config["outdoors"]
    hz_ctl = sample_set_config["simulation"]["hz_ctl"]
    hz_sim = sample_set_config["simulation"]["hz_sim"]
    t_dly = sample_set_config["simulation"]["delay"]

    # Save Path
    output_path = os.path.join(workspace_path,"cohorts",cohort,"output")
    if os.path.isfile(output_path):
        pass
    else:
        # Create Folder/File
        os.makedirs(output_path,exist_ok=True)

    # Overarching Parameters
    base_drone_config = qc.generate_preset_config(drone_config)
    Tpi, CPi = ms.solve(course_config)

    # Debug using 3D Plot and State Space Plot
    if method == "alpha":
        ps.CP_to_3D([Tpi],[CPi],n=60)
        ps.CP_to_xv([Tpi],[CPi])
        # print("Course Parameters:\n",np.around(CPi[:,:,0].T,3))

    tXUi = th.ts_to_tXU(Tpi,CPi,base_drone_config,hz_ctl)
    obj,tf = th.ts_to_obj(Tpi,CPi),tXUi[0,-1]
    expert_base = VehicleRateMPC(course_config,base_drone_config,hz_ctl)
    transform = Resize((720, 1280), antialias=True)

    # Append Expert to Roster
    roster = ["expert"]+roster

    # Delete the generated code
    expert_base.clear_generated_code()

    # Sample Set Parameters
    Drones = gd.generate_drones(
        Nsps,drone_set_config,base_drone_config)
    Perturbations = gd.generate_perturbations(
        np.zeros(Nsps),
        trajectory_set_config,
        Tpi,CPi)
    
    # Simulate
    for pilot in roster:
        print("---------------------------------------------------------------------")
        print("Simulating Pilot: ",pilot)

        # Some useful paths
        trajectory_path = os.path.join(output_path,"sim_data_"+course_name+"_"+pilot+".pt")
        video_path = os.path.join(output_path, "sim_video_"+course_name+"_"+pilot+".mp4")

        trajectories = []
        for (drone,perturbation) in zip(Drones,Perturbations):
            t0,x0 = perturbation["t0"],perturbation["x0"]
            expert = VehicleRateMPC(course_config,drone,hz_ctl)
            simulator = expert.generate_simulator(hz_sim)

            # Load Pilot
            if pilot == "expert":
                policy = expert
            else:
                policy = Pilot(cohort,pilot)
                policy.set_mode('deploy')

            # Simulate
            Tro,Xro,Uro,images,Tsol,Adv = simulate_flight(policy,simulator,
                                                       t0,tf,x0,obj,nerf,hz_sim,
                                                       mu_md=mu_md,std_md=std_md,
                                                       mu_sn=mu_sn,std_sn=std_sn,
                                                       t_dly=t_dly,outdoors=outdoors)
        
            # Save Trajectory
            trajectory = {
                "Tro":Tro,"Xro":Xro,"Uro":Uro,
                "Xid":tXUi[1:11,:],"obj":obj,"Ndata":Uro.shape[1],"Tsol":Tsol,"Adv":Adv,
                "rollout_id":"acados_rollout",
                "course":course_name,"drone":drone_name,"method":method}

            trajectories.append(trajectory)

            # Delete the generated code
            expert.clear_generated_code()

        # Save Flight Record if not expert and request. Save both in just video otherwise.
        if use_fr:
            # Save Flight Record
            flight_record = rf.FlightRecorder(
                Xro.shape[0],Uro.shape[0],
                20,tXUi[0,-1],[360,640,3],obj,cohort,course_name,policy.name)
            flight_record.simulation_import(images,Tro,Xro,Uro,tXUi,Tsol,Adv)
            flight_record.save()
        else:
            # Print some diagnostics
            print("Simulated",len(Drones),"[Trajectory+Drone] rollouts.")

            # Save Trajectories
            torch.save(trajectories,trajectory_path)

            # Save video on last trajectory
            images_vd = torch.zeros((images.shape[0],720,1280,3))
            images = torch.from_numpy(images)
            for j in range(images.shape[0]-1):
                img_vd = images[j,:,:,:]
                img_vd = torch.movedim(img_vd,2,0)
                img_vd = transform(img_vd)
                img_vd = torch.movedim(img_vd,0,2)
                images_vd[j,:,:,:] = img_vd
            write_video(video_path,images_vd,expert.hz)

def simulate_flight(controller:Union[VehicleRateMPC,Pilot],simulator:AcadosSimSolver,
                   t0:float,tf:int,x0:np.ndarray,
                   obj:np.ndarray,nerf:nf.NeRF,
                   hz_sim:int=100,
                   mu_md:np.ndarray=np.zeros(10),std_md:np.ndarray=np.zeros(10),
                   mu_sn:np.ndarray=np.zeros(10),std_sn:np.ndarray=np.zeros(10),
                   t_dly:float=0.0,
                   nx:int=10,nu:int=4,
                   outdoors:bool=False
                   ):
    
    # Simulation Variables
    dt = np.round(tf-t0)
    Nsim = int(dt*hz_sim)
    Nctl = int(dt*controller.hz)
    n_sim2ctl = int(hz_sim/controller.hz)
    n_delay = int(t_dly*hz_sim)

    # Scale model noise according to simulation frequency
    mu_md *= 1/n_sim2ctl
    std_md *= 1/n_sim2ctl

    # Rollout Variables
    Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((nx,Nctl+1)),np.zeros((nu,Nctl))
    Imgs = np.zeros((Nctl,nerf.height,nerf.width,nerf.channels),dtype=np.uint8)
    Xro[:,0] = x0

    # Diagnostics Variables
    Tsol = np.zeros((4,Nctl))
    Adv = np.zeros((nu,Nctl))
    
    # Transient Variables
    xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
    ucm = np.array([-0.5,0.0,0.0,0.0])
    udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
    zcr = torch.zeros(controller.model.Nz) if isinstance(controller,Pilot) else None

    # Rollout
    for i in range(Nsim):
        # Get current time and state
        tcr = t0+i/hz_sim

        # Control
        if i % n_sim2ctl == 0:
            # Get current image
            icr = nerf.render(xcr)

            # Add sensor noise and syncronize estimated state
            if outdoors:
                xsn += np.random.normal(loc=mu_sn,scale=std_sn)

                xsn[0:3] = (1.0*xsn[0:3] + 0.0*xcr[0:3])/2
                xsn[3:6] = (0.7*xsn[3:6] + 0.3*xcr[3:6])/2
                xsn[6:10] = (0.3*xsn[6:10] + 0.7*xcr[6:10])/2
            else:
                xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
            xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

            # Generate controller command
            ucm,zcr,adv,tsol = controller.control(ucm,tcr,xsn,obj,icr,zcr)

            # Update delay buffer
            udl[:,0] = udl[:,1]
            udl[:,1] = ucm

        # Extract delayed command
        uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

        # Simulate both estimated and actual states
        xcr = simulator.simulate(x=xcr,u=uin)
        if outdoors:
            xsn = simulator.simulate(x=xsn,u=uin)

        # Add model noise
        xcr = xcr + np.random.normal(loc=mu_md,scale=std_md)
        xcr[6:10] = th.obedient_quaternion(xcr[6:10],xpr[6:10])

        # Update previous state
        xpr = xcr
        
        # Store values
        if i % n_sim2ctl == 0:
            k = i//n_sim2ctl

            Imgs[k,:,:,:] = icr
            Tro[k] = tcr
            Xro[:,k+1] = xcr
            Uro[:,k] = ucm
            Tsol[:,k] = tsol
            Adv[:,k] = adv

    # Log final time
    Tro[Nctl] = t0+Nsim/hz_sim

    return Tro,Xro,Uro,Imgs,Tsol,Adv