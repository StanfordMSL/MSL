"""
Helper functions for trajectory data.
"""

import numpy as np
import math
from scipy.spatial.transform import Rotation
from typing import Dict,Union

def fo_to_xu(fo:np.ndarray,quad:Dict[str,Union[float,np.ndarray]]):
    """
    Converts a flat output vector to a state vector and control
    input (normalized motor forces).

    Args:
        fo:     Flat output vector.
        m:      Quadcopter mass.
        I:      Quadcopter inertia matrix.
        wMf:    Quadcopter motor force matrix.

    Returns:
        xut:    State vector and control input.
    """

    # Unpack
    pt = fo[0:3,0]
    vt = fo[0:3,1]
    at = fo[0:3,2]
    jt = fo[0:3,3]

    psit  = fo[3,0]
    psidt = fo[3,1]

    m,k_th = quad["m"],quad["fn"]

    # Compute Gravity
    gt = np.array([0.00,0.00,-9.81])

    # Compute Thrust
    alpha:np.ndarray = at+gt

    # Compute Intermediate Frame xy
    xct = np.array([ np.cos(psit), np.sin(psit), 0.0 ])
    yct = np.array([-np.sin(psit), np.cos(psit), 0.0 ])
    
    # Compute Orientation
    xbt = np.cross(alpha,yct)/np.linalg.norm(np.cross(alpha,yct))
    ybt = np.cross(xbt,alpha)/np.linalg.norm(np.cross(xbt,alpha))
    zbt = np.cross(xbt,ybt)
    
    Rt = np.hstack((xbt.reshape(3,1), ybt.reshape(3,1), zbt.reshape(3,1)))
    qt = Rotation.from_matrix(Rt).as_quat()

    # Compute Thrust Variables
    c = zbt.T@alpha

    # Compute Angular Velocity
    B1 = c
    D1 = xbt.T@jt
    A2 = c
    D2 = -ybt.T@jt
    B3 = -yct.T@zbt
    C3 = np.linalg.norm(np.cross(yct,zbt))
    D3 = psidt*(xct.T@xbt)

    wxt = (B1*C3*D2)/(A2*(B1*C3))
    wyt = (C3*D1)/(B1*C3)
    wzt = ((B1*D3)-(B3*D1))/(B1*C3)

    wt = np.array([wxt,wyt,wzt])

    # Compute Body-Rate Command
    ut = np.hstack((m*c*k_th/4,wt))

    # Stack
    xut = np.hstack((pt,vt,qt,ut))

    return xut

def ts_to_fo(tk:float,tf:float,CP:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory spline (defined by tf,CP) to a flat output.

    Args:
        tk:     Current time.
        tf:     Trajectory final time.
        CP:     Control points.

    Returns:
        fo:     Flat output vector.
    """
    Ncp = CP.shape[1]
    M = get_M(Ncp)

    fo = np.zeros((4,Ncp))
    for i in range(0,Ncp):
        nt = get_nt(tk,tf,i,Ncp)
        fo[:,i] = (CP @ M @ nt) / (tf**i)

    return fo

def ts_to_xu(tk:float,tf:float,CP:np.ndarray,
             quad:Dict[str,Union[float,np.ndarray]]) -> np.ndarray:
    """
    Converts a trajectory spline (defined by tf,CP) to a state vector and control input.

    Args:
        tk:     Current segment time.
        tf:     Trajectory segment final time.
        CP:     Control points.
        quad:   Quadcopter configuration.

    Returns:
        xu:    State vector and control input.
    """
    fo = ts_to_fo(tk,tf,CP)
    return fo_to_xu(fo,quad)

def TS_to_xu(tk:float,Tps:np.ndarray,CPs:np.ndarray,
             quad:Dict[str,Union[float,np.ndarray]]) -> np.ndarray:
    """
    Finds the state vector and control input at a given time within a sequence of
    trajectory splines.

    Args:
        tk:     Current time.
        Tps:    Trajectory segment times.
        CPs:    Trajectory control points.
        quad:   Quadcopter configuration.

    Returns:
        xu:    State vector and control input.
    """
    idx = np.where(Tps < tk)[0][-1]
    tf_sm = Tps[idx+1]-Tps[idx]
    tk_sm = tk-Tps[idx]
    CP_sm = CPs[idx,:,:]

    fo = ts_to_fo(tk_sm,tf_sm,CP_sm)

    return fo_to_xu(fo,quad)

def ts_to_tXU(Tps:np.ndarray,CPs:np.ndarray,
              quad:Union[None,Dict[str,Union[float,np.ndarray]]],
              hz:int) -> np.ndarray:
    """
    Converts a trajectory spline (defined by tf,CP) to a trajectory rollout.

    Args:
        Tps:     Trajectory segment times.
        CPs:     Trajectory control points.
        quad:   Quadcopter configuration.
        hz:     Control loop frequency.

    Returns:
        tXU:    State vector and control input rollout.
    """
    Nt = int((Tps[-1]-Tps[0])*hz+1)

    idx = 0
    for k in range(Nt):
        tk = Tps[0]+k/hz

        if tk > Tps[idx+1] and idx < len(Tps)-2:
            idx += 1

        t0,tf = Tps[idx],Tps[idx+1]
        CPk = CPs[idx,:,:]
        xu = ts_to_xu(tk,tf-t0,CPk,quad)

        if k == 0:
            ntxu = len(xu)+1
            tXU = np.zeros((ntxu,Nt))
        else:
            xu[6:10] = obedient_quaternion(xu[6:10],tXU[7:11,k-1])
                
        tXU[0,k] = tk
        tXU[1:,k] = xu

    return tXU

def ts_to_obj(Tp:np.ndarray,CP:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory spline to an objective vector.

    Args:
        Tp:     Trajectory segment times.
        CP:     Trajectory control points.

    Returns:
        obj:    Objective vector.
    """
    tXU = ts_to_tXU(Tp,CP,None,1)

    return tXU_to_obj(tXU)

def tXU_to_obj(tXU:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory rollout to an objective vector.

    Args:
        tXU:    Trajectory rollout.

    Returns:
        obj:    Objective vector.
    """

    dt = tXU[0,-1]-tXU[0,0]
    dp = tXU[1:4,-1]-tXU[1:4,0]
    v0,v1 = tXU[4:7,0],tXU[4:7,-1]
    q0,q1 = tXU[7:11,0],tXU[7:11,-1]
    
    obj = np.hstack((dt,dp,v0,v1,q0,q1)).reshape((-1,1))

    return obj

def xu_to_fo(xuv:np.ndarray,quad:Dict[str,Union[float,np.ndarray]]) -> np.ndarray:
    """
    Converts a state vector to approximation of flat output vector.

    Args:
        xuv:     State vector.

    Returns:
        fo:     Flat output vector.
    """

    # Unpack variables
    wxk,wyk,wzk = xuv[10],xuv[11],xuv[12]
    m,I,fMw = quad["m"],quad["I"],quad["fMw"]

    # Initialize output
    fo = np.zeros((4,5))

    # Compute position terms
    fo[0:3,0] = xuv[0:3]

    # Compute velocity terms
    fo[0:3,1] = xuv[3:6]

    # Compute acceleration terms
    Rk = Rotation.from_quat(xuv[6:10]).as_matrix()      # Rotation matrix
    xbt,ybt,zbt = Rk[:,0],Rk[:,1],Rk[:,2]               # Body frame vectors
    gt = np.array([0.00,0.00,-9.81])                    # Acceleration due to gravity vector
    c = (fMw@xuv[13:17])[0]/m                           # Acceleration due to thrust vector

    fo[0:3,2] = c*zbt-gt

    # Compute yaw term
    psi = np.arctan2(Rk[1,0], Rk[0,0])

    fo[3,0]  = psi

    # Compute yaw rate term
    xct = np.array([np.cos(psi), np.sin(psi), 0])     # Intermediate frame x vector
    yct = np.array([-np.sin(psi), np.cos(psi), 0])    # Intermediate frame y vector
    B1 = c
    B3 = -yct.T@zbt
    C3 = np.linalg.norm(np.cross(yct,zbt))
    D1 = wyk*(B1*C3)/C3
    D3 = (wzk*(B1*C3)+(B3*D1))/B1

    psid = D3/(xct.T@xbt)

    fo[3,1] = psid

    # Compute yaw acceleration term
    Iinv = np.linalg.inv(I)
    rv1:np.ndarray = xuv[10:13]            # intermediate variable
    rv2:np.ndarray = I@xuv[10:13]          # intermediate variable
    utau = (fMw@xuv[13:17])[1:4]
    wd = Iinv@(utau - np.cross(rv1,rv2))
    E1 = wd[1]*(B1*C3)/C3
    E3 = (wd[2]*(B1*C3)+(B3*E1))/B1

    psidd = (E3 - 2*psid*wzk*xct.T@ybt + 2*psid*wyk*xct.T@zbt + wxk*wyk*yct.T@ybt + wxk*wzk*yct.T@zbt)/(xct.T@xbt)

    fo[3,2] = psidd

    return fo

def get_nt(tk:float,tf:float,kd:int,Ncp:int):  
    """
    Generates the normalized time vector based on desired derivative order.

    Args:
        tk:     Current time.
        tf:     Trajectory final time.
        kd:     Derivative order.
        Ncp:    Number of control points.

    Returns:
        nt:      the normalized time vector.
    """

    tn = tk/tf

    nt = np.zeros(Ncp)
    for i in range(kd,Ncp):
        c = math.factorial(i)/math.factorial(i-kd)
        nt[i] = c*tn**(i-kd)
    
    return nt

def get_M(Ncp:int):
    """
    Generates the M matrix for polynomial interpolation.

    Returns:
        M:      Polynomial interpolation matrix.
    """
    M = np.zeros((Ncp,Ncp))
    for i in range(Ncp):
        ci = (1/(Ncp-1))*i
        for j in range(Ncp):
            M[i,j] = ci**j
    M = np.linalg.inv(M).T

    return M

def obedient_quaternion(qcr:np.ndarray,qpr:np.ndarray) -> np.ndarray:
    """
    Ensure that the quaternion is well-behaved (unit norm and closest to reference).
    
    Args:
        qcr:    Current quaternion.
        qpr:    Previous quaternion.

    Returns:
        qcr:     Closest quaternion to reference.
    """
    qcr = qcr/np.linalg.norm(qcr)

    if np.dot(qcr,qpr) < 0:
        qcr = -qcr

    return qcr