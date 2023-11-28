# Isentropic Euler: p = C*rho^gamma, momentum is 'rhouu + p'
# Isothermal Euler: T = cste, momentum is 'rhouu + rho a^2'
import numpy as np
from math import sqrt
from alert import *


# Constants indicating variables position in the unknown vector q
i_rho = 0
i_rhou = 1
i_rhov = 2


def compute_time_step(mesh, q, dt, a, CFL, local):
    """
    Compute time step respecting CFL condition. This function fills the `dt`
    array, it doesn't return anything.

    `q` is a matrix of size (ncells, 3) containing for each cell the values of the
    conservatives variables.

    `dt` is a vector (!WARNING!) containing the time step for each cell.

    `a` is the sound velocity (scalar), `CFL` is the CFL value to respect (scalar)

    `local` is a boolean indicating if a local time step should be applied or not.

    Computing the CFL criteria necessitates a characteristic length of the cell.
    For simplicity, we will use the length of the shortest edge of each cell as
    the characteristic length.

    Note that this information (shortest edge length for each cell) could be stored
    in mesh.properties instead of being recomputed here at each time step
    """
    L = 1e9
    for icell in range(mesh.ncells()):
        for iface in mesh.c2f[icell]:
            L = min(L, mesh.face_area(iface))

        u = np.linalg.norm(q[icell, 1::]/q[icell, 0])
        dt[icell] = CFL * L / max(abs(u - a), abs(u + a))

    if not local :
        dt[:] = np.min(dt)



def center_flux(qL, qR, a):
    """
    Apply pure center flux (wrong for hyperbolic !)
    """
    return euler_isothermal_flux(0.5*(qL + qR), a)


def roe_euler_isothermal_1D3(qL, qR, a):
    """
    Roe flux for Euler isothermal equations (1D, 3 components)
    """
    # Unpack
    rhoL, rhouL, rhovL = qL
    rhoR, rhouR, rhovR = qR

    # Velocities
    uL = rhouL / rhoL
    uR = rhouR / rhoR
    vL = rhovL / rhoL
    vR = rhovR / rhoR

    # Mean value
    ut = (sqrt(rhoL) * uL + sqrt(rhoR) * uR) / (sqrt(rhoL) + sqrt(rhoR))
    vt = (sqrt(rhoL) * vL + sqrt(rhoR) * vR) / (sqrt(rhoL) + sqrt(rhoR))

    # Eigenvalues (abs values)
    lambda1 = abs(ut - a)
    lambda2 = abs(ut)
    lambda3 = abs(ut + a)

    # Eigenvectors
    K1 = np.array([1, ut - a, vt])
    K2 = np.array([0, 0, 1])
    K3 = np.array([1, ut + a, vt])

    # Wave strength
    alpha1 = ((ut + a) * (rhoR - rhoL) - (rhouR - rhouL)) / (2 * a)
    alpha2 = (rhovR - rhovL) - vt * (rhoR - rhoL)
    alpha3 = (-(ut - a) * (rhoR - rhoL) + (rhouR - rhouL)) / (2 * a)

    # Assemble
    F = 0.5 * (
        euler_isothermal_flux(qL, a)
        + euler_isothermal_flux(qR, a)
        - alpha1 * lambda1 * K1
        - alpha2 * lambda2 * K2
        - alpha3 * lambda3 * K3
    )

    return F


def euler_isothermal_flux(q, a):
    """
    Compute the flux associated to the 1D isothermal Euler equations with two velocity components.

    `q` is a array of size 3 containing the conservative quantities. `a` the sound velocity.

    The expected result is the flux vector.
    """
    Fx = np.array([q[1],q[1]**2/q[0]+a**2*q[0],q[1]*q[2]/q[0]])
    #Fy = np.array([q[2],q[1]*q[2]/q[0],q[2]**2/q[0]+a**2*q[0]])
    
    return Fx



def compute_inner_flux(mesh, q, flux, a):
    """
    Compute the flux for each face inside the domain (i.e excluding boundary face). The matrix
    `flux` must be filled (this function returns nothing).

    `q` is a matrix of size (ncells, 3) containing, for each cell, the conservative variables.

    `flux` is a matrix of size (nfaces, 3) that will receive the flux values for each inner face.

    `a` is the sound velocity.
    """
    for i in range(mesh.nfaces()):
        c1,c2 = mesh.f2c[i]
        if c2>=0:
            flux[i,:] = compute_face_flux(mesh.face_normal(i),q[c1],q[c2],a)


    

def compute_face_flux(n, qi, qj, a):
    """
    Compute flux for one specific face
    """

    # Build rotation matrix (and its inverse)
    R = np.array([[1, 0, 0], [0, n[0], n[1]], [0, -n[1], n[0]]])
    Rinv = np.array([[1, 0, 0], [0, n[0], -n[1]], [0, n[1], n[0]]])

    # Rotate state
    rot_qi = R.dot(qi)
    rot_qj = R.dot(qj)

    # Apply 1D numerical flux
    # F = center_flux(rot_qi, rot_qj, a)
    F = roe_euler_isothermal_1D3(rot_qi, rot_qj, a)

    # Rotate back
    F = Rinv.dot(F)

    return F


def boundary_conditions(mesh, q, flux, params):
    """
    Apply the different boundary conditions
    """

    # Loop over boundaries
    for (name, faces) in mesh.bnd2f.items():

        # Loop over faces
        for kface in faces:
            # Get inside cell (index and state)
            icell = mesh.f2c[kface, 0]
            qi = q[icell, :]

            # Normal vector
            n = mesh.face_normal(kface)

            # Apply corresponding condition
            if name == "INLET":
                flux[kface, :] = inlet_condition(n, qi, params)
            elif name == "OUTLET":
                flux[kface, :] = outlet_condition(n, qi, params)
            elif name == "WALL":
                flux[kface, :] = wall_condition(n, qi, params)
            elif name == "FARFIELD":
                flux[kface, :] = farfield_condition(n, qi, params)
            else:
                error("Condition limite '" + name + "' non implémentée")


def inlet_condition(n, qi, params):
    """
    Apply inlet boundary condition.

    Mass flow rate (mfr) is imposed. Density is extrapolated.
    """
    rhouInf = params["rhouInf"]
    rhovInf = params["rhovInf"]
    a = params["a"]

    # Compute ghost state (`g` stands for ghost)
    qg = np.array([qi[0], -rhouInf*n[0], -rhovInf*n[1]])

    # Apply usual flux
    return compute_face_flux(n, qi, qg, a)


def outlet_condition(n, qi, params):
    """
    Apply outlet boundary condition : subsonic outflow with imposed "pressure"
    """
    rhoInf = params["rhoInf"]
    a = params["a"]

    # Compute ghost state (`g` stands for ghost)
    qg = np.array([rhoInf, qi[1]*n[0], qi[2]*n[1]])

    # Apply usual flux
    return compute_face_flux(n, qi, qg, a)


def wall_condition(n, qi, params):
    """
    Apply wall condition

    The procedure uses a ghost cell.

    We want <u_wall,n> = 0. Since u_wall ~ (uL + uR) / 2,
    we have <uR,n> = - <uL,n>.

    Then we don't want to change the velocity in
    the tangential direction, so we set <uR,t>=<uL,t>.

    The same goes for the density, we set rhoR = rhoL

    Since uR = <uR,n>n + <uR,t>t, we have rhoR*uR = rhoL*uL - 2(<rhoL*uL,n>)n
    """
    a = params["a"]

    # Compute ghost state (`g` stands for ghost)
    qg = np.zeros_like(qi)
    qg[0] = qi[0]
    qg[1:] = qi[1:] - 2*np.dot(qi[1:], n)*n

    # Apply usual flux
    return compute_face_flux(n, qi, qg, a)


def farfield_condition(n, qi, params):
    """
    Apply farfield condition

    This condition uses the Riemann problem solution
    to compute a ghost cell state
    """
    # Unpack
    nx, ny = n
    rhoL, rhouL, rhovL = qi

    rhoInf = params["rhoInf"]
    rhouInf = params["rhouInf"]
    rhovInf = params["rhovInf"]
    gamma = params["gamma"]
    a = params["a"]

    c = sqrt(gamma) * a  # sound velocity

    # Compute left state
    unL = (rhouL/rhoL)*nx + (rhovL/rhoL)*ny
    utL = rhouL/rhoL - unL*nx
    vtL = rhovL/rhoL - unL*ny
    pL = rhoL * a**2

    # Upstream state
    Pinf = rhoInf * a**2

    # 2 out of 3 Riemann invariants (independent of velocity sign)
    Rp = rhouInf*nx/rhoInf+rhovInf*ny/rhoInf + 2*c/(gamma-1)
    Rm = unL - 2*c/(gamma-1)

    un_star = 0.5*(Rm+Rp)
    c_star = (gamma-1)*(Rp-Rm)/4

    if (unL > 0):  # IN condition
        unInf = rhouInf*nx/rhoInf + rhovInf*ny/rhoInf
        utInf = rhouInf/rhoInf - unInf*nx
        vtInf = rhovInf/rhoInf - unInf*ny

        R0 = Pinf/(rhoInf**gamma)

        rho_star = (c_star**2/gamma/R0)**(1/(gamma-1))

        rhoR = rho_star
        rhouR = rho_star*(un_star*nx + utInf)
        rhovR = rho_star*(un_star*ny + vtInf)

    else:  # OUT CONDITION
        R0 = pL/(rhoL**gamma)

        rho_star = (c_star**2/gamma/R0)**(1/(gamma-1))

        rhoR = rho_star
        rhouR = rho_star*(un_star*nx + utL)
        rhovR = rho_star*(un_star*ny + vtL)

    # Set ghost state
    qg = np.array([rhoR, rhouR, rhovR])

    # Apply usual flux
    return compute_face_flux(n, qi, qg, a)


def solve_one_time_step(mesh, q, flux, dt, params):
    """
    Solve the Euler isothermal equation for one time step. This function updates
    the values of `q`. It doesn't return anything.

    `q` is a matrix of size (ncells, 3) containing for each cell the values of the
    conservatives variables.

    `flux` is a pre-allocated matrix of size (nfaces, 3) to hold the flux vector values
    at each face

    `dt` is a vector of size (ncells) containing the time step in each cell. It is set
    by this function.
    """
    # Unpack the parameters stored in the dict
    a = params["a"]
    CFL = params["CFL"]
    local = params["localTimeStep"]

    # Compute time step
    compute_time_step(mesh, q, dt, a, CFL, local)
    # Compute flux on inner faces
    compute_inner_flux(mesh, q, flux, a)
    # Apply boundary condition
    boundary_conditions(mesh, q, flux, params)
    # Perform explicit integration
    for i_cell in range(mesh.ncells()):
        for j_face in mesh.c2f[i_cell]:
            if mesh.f2c[j_face,0]==i_cell:
                q[i_cell] -= dt[i_cell]/mesh.cell_volume(i_cell) * flux[j_face] * mesh.face_area(j_face)
            else : 
                q[i_cell] += dt[i_cell]/mesh.cell_volume(i_cell) * flux[j_face] * mesh.face_area(j_face)
    
    
    
    
    
    
    

    
