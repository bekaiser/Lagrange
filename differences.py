# first and second derivatives using Lagrange polynomials
# Bryan Kaiser

import numpy as np
import math as ma
import matplotlib.pyplot as plt

figure_path = './figures/'

# =============================================================================
# functions


def weights2( z0 , z1 , z2 , z3 , zj ):
 # Lagrange polynomial weights for second derivative
 l0 = 1./(z0-z1) * ( 1./(z0-z2) * (zj-z3)/(z0-z3) + 1./(z0-z3) * (zj-z2)/(z0-z2) ) + \
      1./(z0-z2) * ( 1./(z0-z1) * (zj-z3)/(z0-z3) + 1./(z0-z3) * (zj-z1)/(z0-z1) ) + \
      1./(z0-z3) * ( 1./(z0-z1) * (zj-z2)/(z0-z2) + 1./(z0-z2) * (zj-z1)/(z0-z1) )
 l1 = 1./(z1-z0) * ( 1./(z1-z2) * (zj-z3)/(z1-z3) + 1./(z1-z3) * (zj-z2)/(z1-z2) ) + \
      1./(z1-z2) * ( 1./(z1-z0) * (zj-z3)/(z1-z3) + 1./(z1-z3) * (zj-z0)/(z1-z0) ) + \
      1./(z1-z3) * ( 1./(z1-z0) * (zj-z2)/(z1-z2) + 1./(z1-z2) * (zj-z0)/(z1-z0) )
 l2 = 1./(z2-z0) * ( 1./(z2-z1) * (zj-z3)/(z2-z3) + 1./(z2-z3) * (zj-z1)/(z2-z1) ) + \
      1./(z2-z1) * ( 1./(z2-z0) * (zj-z3)/(z2-z3) + 1./(z2-z3) * (zj-z0)/(z2-z0) ) + \
      1./(z2-z3) * ( 1./(z2-z0) * (zj-z1)/(z2-z1) + 1./(z2-z1) * (zj-z0)/(z2-z0) )
 l3 = 1./(z3-z0) * ( 1./(z3-z1) * (zj-z2)/(z3-z2) + 1./(z3-z2) * (zj-z1)/(z3-z1) ) + \
      1./(z3-z1) * ( 1./(z3-z0) * (zj-z2)/(z3-z2) + 1./(z3-z2) * (zj-z0)/(z3-z0) ) + \
      1./(z3-z2) * ( 1./(z3-z0) * (zj-z1)/(z3-z1) + 1./(z3-z1) * (zj-z0)/(z3-z0) )
 return l0, l1, l2, l3


def partial_zz( z ):
 # second derivative, permiting non-uniform grids
 Nz = np.shape(z)[0]
 # 2nd order accurate (truncated 3rd order terms), variable grid
 diagm1 = np.zeros([Nz-1])
 diag0 = np.zeros([Nz])
 diagp1 = np.zeros([Nz-1])
 for j in range(1,Nz-1):
     denom = 1./2. * ( z[j+1] - z[j-1] ) * ( z[j+1] - z[j] ) * ( z[j] - z[j-1] )  
     diagm1[j-1] = ( z[j+1] - z[j] ) / denom
     diagp1[j] =   ( z[j] - z[j-1] ) / denom
     diag0[j] =  - ( z[j+1] - z[j-1] ) / denom
 pzz = np.diag(diagp1,k=1) + np.diag(diagm1,k=-1) + np.diag(diag0,k=0) 
 # lower (wall) BC sets variable to zero at the wall
 zj = z[0] # location of derivative for lower BC (first cell center)
 l0, l1, l2, l3 = weights2( -z[0] , z[0] , z[1] , z[2] , zj )
 l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_1)
 pzz[0,0:3] = [ l1 , l2 , l3 ]
 # upper (far field) BC
 zj = z[Nz-1] # location of derivative for upper BC
 l0, l1, l2, l3 = weights2( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , zj )
 l2 = l3 + l2 # Neumann for phi_z=0 at z=H
 pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 return pzz


def partial_z( z ):
 # first-order derivative matrix 
 # 2nd order accurate truncation
 Nz = np.shape(z)[0]
 
 # interior points, variable grid
 diagm1 = np.zeros([Nz-1])
 diag0 = np.zeros([Nz])
 diagp1 = np.zeros([Nz-1])
 for j in range(1,Nz-1):
   denom = ( ( z[j+1] - z[j] ) * ( z[j] - z[j-1] ) * ( ( z[j+1] - z[j] ) + ( z[j] - z[j-1] ) ) ) 
   diagm1[j-1] = - ( z[j+1] - z[j] )**2. / denom
   diagp1[j] =   ( z[j] - z[j-1] )**2. / denom
   diag0[j] =  ( ( z[j+1] - z[j] )**2. - ( z[j] - z[j-1] )**2. ) / denom
 pz = np.diag(diagp1,k=1) + np.diag(diagm1,k=-1) + np.diag(diag0,k=0) 

 # lower (wall) BC sets variable to zero at the wall
 l0, l1, l2, l3 = weights( -z[0] , z[0] , z[1] , z[2] , z[0] )
 l1 = l1 - l0 # Dirchlet phi=0 at z=0
 pz[0,0:3] = [ l1 , l2 , l3 ]
   
 # upper (far field) BC
 l0, l1, l2, l3 = weights( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , z[Nz-1] )
 l2 = l3 + l2 # Neumann for phi_z=0 at z=H
 pz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 
 return pz


def weights( z0 , z1 , z2 , z3 , zj ):
 # Lagrange polynomial weights for first derivative
 l0 = 1./(z0-z1) * (zj-z2)/(z0-z2) * (zj-z3)/(z0-z3) + \
      1./(z0-z2) * (zj-z1)/(z0-z1) * (zj-z3)/(z0-z3) + \
      1./(z0-z3) * (zj-z1)/(z0-z1) * (zj-z2)/(z0-z2)
 l1 = 1./(z1-z0) * (zj-z2)/(z1-z2) * (zj-z3)/(z1-z3) + \
      1./(z1-z2) * (zj-z0)/(z1-z0) * (zj-z3)/(z1-z3) + \
      1./(z1-z3) * (zj-z0)/(z1-z0) * (zj-z2)/(z1-z2)
 l2 = 1./(z2-z0) * (zj-z1)/(z2-z1) * (zj-z3)/(z2-z3) + \
      1./(z2-z1) * (zj-z0)/(z2-z0) * (zj-z3)/(z2-z3) + \
      1./(z2-z3) * (zj-z0)/(z2-z0) * (zj-z1)/(z2-z1)
 l3 = 1./(z3-z0) * (zj-z1)/(z3-z1) * (zj-z2)/(z3-z2) + \
      1./(z3-z1) * (zj-z0)/(z3-z0) * (zj-z2)/(z3-z2) + \
      1./(z3-z2) * (zj-z0)/(z3-z0) * (zj-z1)/(z3-z1)
 return l0, l1, l2, l3


# =============================================================================
# loop over Nz resolution Chebyshev node grid

max_exp = 15 # power of two, must be equal to or greater than 5 (maximum N = 2^max)
Nr = np.power(np.ones([max_exp-3])*2.,np.linspace(4.,max_exp,max_exp-3)) # resolution 
Ng = int(np.shape(Nr)[0]) # number of resolutions to try

Linf1 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid

Linf2 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 

 
H = 1.0 # domain height

for n in range(0,Ng): 
  
  Nz = int(Nr[n]) # resolution
  print('Number of grid points: ',Nz)
 
  dz = H/Nz
  z = np.linspace(dz, Nz*dz, num=Nz)-dz/2. # uniform grid

  zc = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H
  zc = zc[0:Nz] # half cosine grid
  dzc = zc[1:Nz] - zc[0:Nz-1]

  if n == 0:
   plotname = figure_path + 'uniform_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz,z,'ob',label=r"centers")
   plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}",fontsize=13)
   plt.ylabel(r"$z$",fontsize=13)
   plt.grid()
   plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'cosine_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz,zc,'ob',label=r"centers")
   plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}",fontsize=13)
   plt.ylabel(r"$z$",fontsize=13)
   plt.grid()
   plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

  U0 = 2. # free stream velocity

  m = np.pi/(2.*H)
  u = np.zeros([Nz,1])
  uz = np.zeros([Nz,1])
  uzz = np.zeros([Nz,1])
  u[:,0] = U0*np.sin(m*z) # signal velocity u
  uz[:,0] = U0*m*np.cos(m*z) # du/dz
  uzz[:,0] = -U0*m**2.*np.sin(m*z) # d^2u/dz^2

  uc = np.zeros([Nz,1])
  uzc = np.zeros([Nz,1])
  uzzc = np.zeros([Nz,1])
  uc[:,0] = U0*np.sin(m*zc) 
  uzc[:,0] = U0*m*np.cos(m*zc) 
  uzzc[:,0] = -U0*m**2.*np.sin(m*zc)

  if n == 6:
   plotname = figure_path + 'solutions_uniform_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(u/U0,z,'b',label=r"$u$")
   plt.plot(uz/(m*U0),z,'k',label=r"$u_z$")
   plt.plot(-uzz/(m**2.*U0),z,'--r',label=r"$u_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=6,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'solutions_cosine_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uc/U0,zc,'b',label=r"$u$")
   plt.plot(uzc/(m*U0),zc,'k',label=r"$u_z$")
   plt.plot(-uzzc/(m**2.*U0),zc,'--r',label=r"$u_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=6,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

  #print( partial_z( z , 0 ) )
  #print( partial_z( z , 1 ) )

  # compute 1st derivatives:
  uz0 = np.dot( partial_z( z ) , u ) # uniform grid
  uz0c = np.dot( partial_z( zc ) , uc ) # cosine grid

  # compute 2nd dericatives:
  uzz0 = np.dot( partial_zz( z ) , u ) # uniform grid
  uzz0c = np.dot( partial_zz( zc ) , uc ) # cosine grid

  if n == 6:
   plotname = figure_path + 'computed_1st_derivative_uniform_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uz/(m*U0),z,'k',label=r"analytical")
   plt.plot(uz0/(m*U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   #plt.axis([0.,0.004,0.996,1.00])
   #plt.axis([0.9999,1.0001,0.,0.005])
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_cosine_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uzc/(m*U0),zc,'k',label=r"analytical")
   plt.plot(uz0c/(m*U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);
 
   plotname = figure_path + 'computed_2nd_derivative_uniform_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(-uzz/(m**2.*U0),z,'k',label=r"analytical")
   plt.plot(-uzz0/(m**2.*U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(-uzzc/(m**2.*U0),zc,'k',label=r"analytical")
   plt.plot(-uzz0c/(m**2.*U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_uniform_grid_error.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uz/(m*U0)-uz0/(m*U0)),abs(uz/(m*U0))),z,'k') 
   plt.plot(abs(uz-uz0)/abs(m*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_cosine_grid_error.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uzc/(m*U0)-uz0c/(m*U0)),abs(uzc/(m*U0))),z,'k') 
   plt.plot(abs(uzc-uz0c)/abs(m*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_uniform_grid_error.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uzz/(m*U0)-uzz0/(m*U0)),abs(uzz/(m*U0))),z,'k') 
   plt.plot(abs(uzz-uzz0)/abs(m**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid_error.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uzzc/(m*U0)-uzz0c/(m*U0)),abs(uzzc/(m*U0))),z,'k') 
   plt.plot(abs(uzzc-uzz0c)/abs(m**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);


  Linf1[n] = np.amax(abs(uz-uz0)/abs(m*U0)) 
  Linf1c[n] = np.amax(abs(uzc-uz0c)/abs(m*U0))

  Linf2[n] = np.amax(abs(uzz-uzz0)/abs(m**2.*U0)) 
  Linf2c[n] = np.amax(abs(uzzc-uzz0c)/abs(m**2.*U0))


plotname = figure_path + 'first_derivative_error.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf1,'r',label=r"uniform")
plt.loglog(Nr,Linf1c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1$^{st}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'second_derivative_error.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf2,'r',label=r"uniform")
plt.loglog(Nr,Linf2c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2$^{nd}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


