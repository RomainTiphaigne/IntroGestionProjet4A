#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:38:13 2019

@author: bouclier
"""

import numpy as np 
import scipy.sparse 
from scipy.sparse.linalg import spsolve
import nurbs as nb 
import copy 
import matplotlib.pyplot as plt


 
#%% CLASSE POUR DÉFINIR L'OBJET : COURBE NURBS
################################################

class NURBS_Curve: 
    """ NURBS curve object """ 
    def __init__(self,ctrlPts,pp,xxsi):
        """ Attributs  """ 
        self.ctrlPts=ctrlPts    # Inhomogeneous Control Points Coordinates and weights ( x,y,w)
        self.pp=pp              # degree  of the curve 
        self.xxsi=xxsi          # knot vector 
        self.nel = None         # number of elements 
        self.ien = None         # connectivity table
        self.nen = None         # number of basis functions that supports an element 
        self.nbf = None         # total number of basis functions
            
    def DegElevation(self,ppnew):
        # degree elevation without modifying the initial curve 
        t = ppnew - self.pp 
        if t !=0: 
            self.ctrlPts[:-1,:] = self.ctrlPts[:-1,:]*self.ctrlPts[-1,:] # to homogeneous coordinates !!!! Attention  
            self.ctrlPts, self.xxsi = nb.bspdegelev(self.pp,self.ctrlPts,self.xxsi,t)
            self.pp = ppnew 
            self.ctrlPts[:-1,:] = self.ctrlPts[:-1,:]/self.ctrlPts[-1,:]
    
    def KnotInsertion(self,u_refinement):
        # knot insertion without modifying the initial curve
        if np.size(u_refinement)!=0 : 
            self.ctrlPts[:-1,:] = self.ctrlPts[:-1,:]*self.ctrlPts[-1,:] # to homogeneous coordinates !!!! Attention  
            self.ctrlPts, self.xxsi = nb.bspkntins(self.pp,self.ctrlPts,self.xxsi,u_refinement)
            self.ctrlPts[:-1,:] = self.ctrlPts[:-1,:]/self.ctrlPts[-1,:]
    
    def Connectivity(self):
        self.nbf = self.ctrlPts.shape[1]            # total number of basis functions
        self.nel =  self.nbf - self.pp              # number of elements
        self.ien = nb.nubsconnect(self.pp,self.nbf) # connectivity table
        self.nen = self.pp + 1                      # number of basis functions that supports an element 
     
    def GetBasisFunctionsForGivenPoints(self,xi):
        phi = nb.global_basisfunsPyWd(self.pp,self.xxsi,xi)
        denom      = phi.dot( self.ctrlPts[2,:]   )
        Wphi       = phi.multiply(self.ctrlPts[2,:] ) 
        N = scipy.sparse.diags(1/denom).dot(Wphi)
        return N 
    
    def PlotCurve(self,ax,neval,U=None):
        xi = np.linspace(1.e-8, 1-1.e-8, neval)
        N = self.GetBasisFunctionsForGivenPoints(xi)
        if U is None: 
            xeval = N.dot(self.ctrlPts[0,:])
            yeval = N.dot(self.ctrlPts[1,:])
            ax.plot(self.ctrlPts[0,:],self.ctrlPts[1,:],'o',color='blue')
            ax.plot(self.ctrlPts[0,:],self.ctrlPts[1,:],'-',color='black')
        else: 
            nbf = self.ctrlPts.shape[1]
            x = self.ctrlPts[0,:]+U[:nbf]
            y = self.ctrlPts[1,:]+U[nbf:2*nbf]
            xeval = N.dot(x)
            yeval = N.dot(y)
            ax.plot(x,y,'o',color='blue')
            ax.plot(x,y,'-',color='black')
        ax.plot(xeval,yeval ,'-',color='red')
            
    def Stiffness(self,E,S,mu,I,R):
        
        # SI BESOIN DE DEBUGGER :  import pdb; pdb.set_trace() 
        
        # p+1 Gauss points for the numerical integration
        npg    = self.pp + 1  # number of Gauss points per element 
        xsi_tilde_pg,wpg  =  nb.GaussLegendre(npg) # numpy array collecting Gauss points and weights 
        
        # Initialization
        K = np.zeros([3*self.nbf,3*self.nbf])
        
        # Loop over elements
        for e in range(self.nel):
            
            # On récupère la NURBS coordinate puis on calcule xsii et xsii1
            ni = self.ien[0,e]
            xsii = self.xxsi[ni]
            xsii1 = self.xxsi[ni+1]

            # Measure of the parametric element 
            mes = xsii1 - xsii
            
            # treating only elements of non zero measure 
            if mes > 0 :
                
                # for the connectivity
                # list of non zero global funs over the element
                bf_index = np.sort(self.ien[:,e])
                
                # initialisation of the elementary stiffness matrix Ke_m, Ke_ct, Ke_f
                Ke_m = np.zeros([3*self.nen,3*self.nen])
                Ke_t = np.zeros([3*self.nen,3*self.nen])
                Ke_f = np.zeros([3*self.nen,3*self.nen])
                
                #Loop over Gauss quadrature points 
                for pg in range(npg):
                    
                    # Mapping from parent domain [-1,1] to isoparametric domain [xsii,xsii1]
                    # calculer xsipg en fonction xsi_tilde_pg
                    xsipg = xsii + (xsi_tilde_pg[pg]+1)*((xsii1-xsii)/2)
                    
                    
                    # calculer dxsidxsitilde
                    dxsidxsitilde = (xsii1-xsii)/2
                    
                    # Mapping from isoparametric space to physical space
                    #-------------------------------------------------------
                    
                    # Evaluating the B-spline basis functions (along with derivatives) 
                    phi,dphidxi = nb.derbasisfuns(ni,self.pp,self.xxsi,1,xsipg)
                    
                    # Getting the associated NURBS functions
                    # Calculer les fonctions NURBS N à partir des B-SPline M
                    N = np.zeros(self.nen)
                    SumN = sum(phi*self.ctrlPts[2,bf_index])
                    for i in range(self.nen):
                        N[i] = phi[i]*self.ctrlPts[2,bf_index[i]]/SumN
                    
                    
                    # Calculer la dérivée première des fonctions NURBS
                    dN = np.zeros(self.nen)
                    SumdN = sum(dphidxi*self.ctrlPts[2,bf_index])
                    for i in range(self.nen):
                        dN[i] = ((dphidxi[i]*self.ctrlPts[2,bf_index[i]]*SumN) - (phi[i]*self.ctrlPts[2,bf_index[i]]*SumdN))/SumN**2
                    
                    # Jacobian between physical and isoparametric spaces
                    # Calculer le Jacobien J=dsdxsi
                    dxdxsi = sum(dN*self.ctrlPts[0,bf_index])
                    dydxsi = sum(dN*self.ctrlPts[1,bf_index])
                    J = np.sqrt((dxdxsi**2 + dydxsi**2))
                    
                    # Getting the element stiffness matrices
                    #------------------------------------------------
                    
                    # computation of the dNs for the beam
                    # Calculer les dN poutre
                    dNm = np.zeros(3*self.nen)
                    dNm[:self.nen] = dN/J
                    dNm[self.nen:2*self.nen] = -N/R
                    
                    dNt = np.zeros(3*self.nen)
                    dNt[:self.nen] = N/R
                    dNt[self.nen:2*self.nen] = dN/J
                    dNt[2*self.nen:] = -N
                    
                    dNf = np.zeros(3*self.nen)
                    dNf[2*self.nen:] = dN/J
                     
                    # Integral evaluation
                    # Evaluer l'integrale avec quadrature sur les points de gauss
                    Ke_m += wpg[pg]*E*S*np.outer(dNm,dNm)*J*dxsidxsitilde
                    Ke_t += wpg[pg]*mu*S*np.outer(dNt,dNt)*J*dxsidxsitilde
                    Ke_f += wpg[pg]*E*I*np.outer(dNf,dNf)*J*dxsidxsitilde
                     
                
                # total element stiffness matrix
                Ke = Ke_m + Ke_t + Ke_f
                
                #import pdb; pdb.set_trace()
                
                # Assembling
                for ibf in range(self.nen):
                    for jbf in range(self.nen):
                        # Faire l'assemblage
                        K[bf_index[ibf],bf_index[jbf]] += Ke[ibf,jbf]
                        K[bf_index[ibf],bf_index[jbf]+self.nbf] += Ke[ibf,jbf+self.nen]
                        K[bf_index[ibf],bf_index[jbf]+2*self.nbf] += Ke[ibf,jbf+2*self.nen]
                        
                        K[bf_index[ibf]+self.nbf,bf_index[jbf]] += Ke[ibf+self.nen,jbf]
                        K[bf_index[ibf]+self.nbf,bf_index[jbf]+self.nbf] += Ke[ibf+self.nen,jbf+self.nen]
                        K[bf_index[ibf]+self.nbf,bf_index[jbf]+2*self.nbf] += Ke[ibf+self.nen,jbf+2*self.nen]          
                        
                        K[bf_index[ibf]+2*self.nbf,bf_index[jbf]] += Ke[ibf+2*self.nen,jbf]
                        K[bf_index[ibf]+2*self.nbf,bf_index[jbf]+self.nbf] += Ke[ibf+2*self.nen,jbf+self.nen]
                        K[bf_index[ibf]+2*self.nbf,bf_index[jbf]+2*self.nbf] += Ke[ibf+2*self.nen,jbf+2*self.nen]
                        
        return K
        
    def Rhs(self,h,R):
    
        # p+1 Gauss points for the numerical integration
        npg    = self.pp + 1  # number of Gauss points per element 
        xsi_tilde_pg,wpg  =  nb.GaussLegendre(npg) # numpy array of Gauss points and weights 
        
        # Initialization
        F = np.zeros(3*self.nbf)
        
        # Loop over elements
        for e in range(self.nel):
            
            # !!! On récupère la NURBS coordinate puis on calcule xsii et xsii1
            ni = self.ien[0,e]
            xsii = self.xxsi[ni]
            xsii1 = self.xxsi[ni+1]

            # Measure of the parametric element 
            mes = xsii1 - xsii
            
            # treating only elements of non zero measure 
            if mes > 0 :
                
                # for the connectivity
                # list of non zero global funs over the element
                bf_index = np.sort(self.ien[:,e])
                
                # !!! initialisation of the elementary force vector Fe
                Fe = np.zeros(3*self.nen)
                
                #Loop over Gauss quadrature points 
                for pg in range(npg):
                    
                    # Mapping from parent domain [-1,1] to isoparametric domain [xi_min,xi_max]
                    # !!! calculer xsipg en fonction xsi_tilde_pg
                    xsipg = xsii + (xsi_tilde_pg[pg]+1)*((xsii1-xsii)/2)
                    # !!! calculer dxsidxsitilde
                    dxsidxsitilde = (xsii1-xsii)/2
                    
                    # Mapping from isoparametric space to physical space
                    #-------------------------------------------------------
                    
                    # Evaluating the B-spline basis functions (along with derivatives) 
                    phi,dphidxi = nb.derbasisfuns(ni,self.pp,self.xxsi,1,xsipg)
                    
                    # Getting the associated NURBS functions
                    # !!! Calculer les fonctions NURBS N à partir des B-SPline M
                    N = np.zeros(self.nen)
                    SumN = sum(phi*self.ctrlPts[2,bf_index])
                    for i in range(self.nen):
                        N[i] = phi[i]*self.ctrlPts[2,bf_index[i]]/SumN
                        
                    # !!! Calculer la dérivée première des fonctions NURBS
                    dN = np.zeros(self.nen)
                    SumdN = sum(dphidxi*self.ctrlPts[2,bf_index])
                    for i in range(self.nen):
                        dN[i] = ((dphidxi[i]*self.ctrlPts[2,bf_index[i]]*SumN) - (phi[i]*self.ctrlPts[2,bf_index[i]]*SumdN))/SumN**2
                    
                    # Jacobian between physical and isoparametric spaces
                    # !!! Calculer le Jacobien J=dsdxsi
                    dxdxsi = sum(dN*self.ctrlPts[0,bf_index])
                    dydxsi = sum(dN*self.ctrlPts[1,bf_index])
                    J = np.sqrt((dxdxsi**2 + dydxsi**2))
                    
                    # Getting the element force vector
                    #------------------------------------------------
                    
                    # evaluation of the moment at the Gauss point in the pysical space
                    # !!! calculer l'angle teta du point de Gauss dans le repère physique 
                    y = sum(N*self.ctrlPts[1,bf_index])
                    sin_teta = y/R
                    # !!! puis c(teta)
                    c = h**3 * sin_teta
                    
                    # computation of the dF
                    dF = np.zeros(3*self.nen)
                    dF[2*self.nen:] = N*c
                    
                    # Integral evaluation
                    Fe += wpg[pg]*dF*J*dxsidxsitilde

                
                # Assembling
                for ibf in range(self.nen):   
                    F[bf_index[ibf]] += Fe[ibf]
                    F[bf_index[ibf]+self.nbf] += Fe[ibf+self.nen]
                    F[bf_index[ibf]+2*self.nbf] += Fe[ibf+2*self.nen]
                        
        return F
    
    
    def Err_L2(self,E,R,b,U):
         
        
        # 9 Gauss points for the numerical integration
        npg    = 9  # number of Gauss points per element 
        xsi_tilde_pg,wpg  =  nb.GaussLegendre(npg) # numpy array collecting Gauss points and weights 
        
        # Initialization
        Err_un = 0
        Errex_un = 0
        
        # Loop over elements
        for e in range(self.nel):
            
            # !!! On récupère la NURBS coordinate puis on calcule xsii et xsii1
            ni = self.ien[0,e]
            xsii = self.xxsi[ni]
            xsii1 = self.xxsi[ni+1]

            # Measure of the parametric element 
            mes = xsii1 - xsii
            
            # treating only elements of non zero measure 
            if mes > 0 :
                
                # for the connectivity
                # list of non zero global funs over the element
                bf_index = np.sort(self.ien[:,e])
                
                # getting the local dof vector Ue
                # !!! Récupérer le depl élémentaire Ue associé
                
                #Loop over Gauss quadrature points 
                for pg in range(npg):
                    
                    xsipg = xsii + (xsi_tilde_pg[pg]+1)*((xsii1-xsii)/2)
                    dxsidxsitilde = (xsii1-xsii)/2
                    phi,dphidxi = nb.derbasisfuns(ni,self.pp,self.xxsi,1,xsipg)
                    
                    N = np.zeros(self.nen)
                    SumN = sum(phi*self.ctrlPts[2,bf_index])
                    for i in range(self.nen):
                        N[i] = phi[i]*self.ctrlPts[2,bf_index[i]]/SumN
                        
                    dN = np.zeros(self.nen)
                    SumdN = sum(dphidxi*self.ctrlPts[2,bf_index])
                    for i in range(self.nen):
                        dN[i] = ((dphidxi[i]*self.ctrlPts[2,bf_index[i]]*SumN) - (phi[i]*self.ctrlPts[2,bf_index[i]]*SumdN))/SumN**2
                    
                    dxdxsi = sum(dN*self.ctrlPts[0,bf_index])
                    dydxsi = sum(dN*self.ctrlPts[1,bf_index])
                    J = np.sqrt((dxdxsi**2 + dydxsi**2))
                    
                    y = sum(N*self.ctrlPts[1,bf_index])
                    teta = np.arcsin(y/R)
                    
                    # !!! à vous de jouer pour calculer l'erreur (au carré) (cf ci-dessous)
                    ## A terminer le calcul du U numérique
                    un_e = 0
                    for i in range(self.nen):
                        un_i = U[self.nbf + bf_index[i]]
                        un_e += N[i]*un_i
                    un_ex = 12*R**3/(E*b) * ( 0.5*teta*np.sin(teta) + 0.25*((np.sin(teta)*np.sin(2*teta)+np.cos(teta)*np.cos(2*teta))-np.cos(teta)))
    
                    # error
                    Err_un += wpg[pg]*(un_e-un_ex)**2*J*dxsidxsitilde
                    Errex_un += wpg[pg]*un_ex**2*J*dxsidxsitilde
                    

                    
        # computation of the relative error
        Err_un_rel = (Err_un/Errex_un)**0.5
            
        return Err_un_rel
        
        
                     
#%% FUN POUR DEFINIR OBJET NURBS : CAS QUART DE CERCLE
#######################################################    
def QuarterCircleCurve(R,p,nel,cont):
    """ Returns a Nurbs curve object 
        Exact quarter circle 
        Radius : R 
        P: final degree of the exact circle 
        nel: number of elements of the parametric space of the curve  
    """
    # initial coarse mesh (p=2, nel=1)
    #-------------------------------------
    # cntrl points
    ctrlPts = np.array([[R,R,0],
                        [0,R,R],
                        [1,1/np.sqrt(2),1]])
    # knot vector
    xxsi = np.array([0,0,0,1,1,1])
    # build the initial nurbs object
    ci = NURBS_Curve(ctrlPts,2,xxsi); 
    
    # refined nurbs mesh (p, nel)
    #-------------------------------------
    cnew = copy.deepcopy(ci)
    # degree elevation (with the method DegElevation of the Nurbs_Curve class)
    cnew.DegElevation(p)
    # knot insertion (with the method KnotInsertion of the Nurbs_Curve class)
    if nel > 1 :
        if cont == 'CP1' :
            ubar = np.linspace(0,1,nel+1)[1:-1] # list of knots to inserted
        elif cont == 'C0' :
            ubar = np.linspace(0,1,nel+1)[1:-1]
            ubar = [ i for i in ubar for j in range(p) ]
        else : 
            raise ValueError('Error continuity')
        cnew.KnotInsertion(ubar)
    # build the connectivity of the refined nurs mesh
    # (à l'aide de la methode Connectivity de la classe Nurbs_Curve)
    cnew.Connectivity()
    return cnew,ci 


 

#%% PARAMÈTRES DU PB
################################################
    
# Geometrie
#----------------
# ligne moyenne : rayon de l'arc de cercle
R = 100
# largeur de la poutre
b = 1
# epaisseur de la poutre
h = 10
# section  
S = b*h
# Moment d'inertie de la section
I = (b*h**3)/12 

# Materiau
#----------------
# Module de Young
E = 10000
# coefficient de poisson
nu = 0 
# module de cisaillement
mu = E/(2*(1+nu))

# Chargement
#------------------------
# moment lineique

# CL
#------------------------
# poutre encastree a gauche : imposees par substitution

# Discrétisation
#-------------------------
# nombre d'elements
nel = 10
# degré des fonctions de forme
p = 2
# continuité des fonctions de forme
# cont = 'C0' # continuité C0 (idem EF)
cont = 'CP1' # continuité C^(p-1) (interet IGA)



#%% CONSTRUCTION MAILLAGE NURBS
################################################

# cas du quart de cercle
# !! MARCHE QUE POUR DES ELTS QUADRATIQUE POUR L'INSTANT
# !! COMPLETER LA MÉTHODE QuarterCircleCurve POUR Q5
craf, cinit = QuarterCircleCurve(R,p,nel,cont) 

# Plot courbe (à l'aide de la methode PlotCurve de la classe Nurbs_Curve)
fig, ax = plt.subplots(figsize=(6,6))
cinit.PlotCurve(ax,200)
plt.title('Initial nurbs curve')
plt.grid()
fig, ax = plt.subplots(figsize=(6,6))
craf.PlotCurve(ax,200)
plt.title('Refined nurbs curve')
plt.grid()

#%% CALCUL DE LA MATRICE DE RIGIDITE IGA
#############################################
# à l'aide la methode Stiffness de la classe Nurbs_Curve
# !! COMPLÉTER LA MÉTHODE Stiffness DE LA CLASSE NURBS_curve CI-DESSUS POUR Q6
K = craf.Stiffness(E,S,mu,I,R)


#%% CALCUL DU SECOND MEMBRE IGA
#######################################################
# !! COMPLÉTER LA MÉTHODE Rhs DE LA CLASSE NURBS_curve CI-DESSUS POUR Q7
F = craf.Rhs(h,R)



#%% IMPOSITION DES CL DE DIRICHLET
#######################################################

for i in range(3):
    K[i*craf.nbf,:] = 0
    K[:,i*craf.nbf] = 0
    K[i*craf.nbf,i*craf.nbf] = 1
    F[i*craf.nbf] = 0

    

#%% RESOLUTION DU SYSTÈME LINÉAIRE 
#######################################################

U = spsolve(K,F)


#%% POST-TRAITEMENT 
#######################################################

# amplitude solution en bout de poutre 
#----------------------------------------
# solution théorique en s=L
tetaL = np.pi/2 ;
Ubth = (12*R**2)/(E*b)*np.sin(tetaL);
Unth = (12*R**3)/(E*b)*(tetaL*np.sin(tetaL)/2+np.sin(tetaL)*np.sin(2*tetaL)/4+np.cos(tetaL)*np.cos(2*tetaL)/4-np.cos(tetaL)/4);
Utth = -(12*R**3)/(E*b)*(-3/4*np.sin(tetaL)+tetaL*np.cos(tetaL)/2+np.cos(tetaL)*np.sin(2*tetaL)/2-np.sin(tetaL)*np.cos(2*tetaL)/4);

print('Rotation en bout de poutre theorique Teta_b(L) : ',Ubth)
print('Rotation en bout de poutre Teta_b(L) : ',U[-1])
print('Deplacement tangent en bout de poutre theorique u_t(L) :',Utth)
print('Deplacement tangent en bout de poutre u_t(L) : ',U[craf.nbf-1])
print('Deplacement normal en bout de poutre theorique u_n(L) : ',Unth)
print('Deplacement normal en bout de poutre u_n(L) : ',U[2*craf.nbf-1])

# calcul des erreurs le long de la poutre
#--------------------------------------------
# !! COMPLÉTER LA MÉTHODE Err_L2 LA CLASSE NURBS_curve CI-DESSUS POUR Q10
Err_un = craf.Err_L2(E,R,b,U)

print('Erreur L2 en un : ',Err_un)




    
            