# high-order derivatives using Lagrange polynomials (Fornberg 1998 algorithm)
# Bryan Kaiser

# maybe I should add more forward backward differences to 
# boundaries for improved accuracy (communicate BCs deeper)?

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


def partial_zz( z , lower_BC_flag , upper_BC_flag ):
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
 if lower_BC_flag == 'dirchlet':
   l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
 if lower_BC_flag == 'neumann':
   l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
 pzz[0,0:3] = [ l1 , l2 , l3 ]

 # upper (far field) BC
 zj = z[Nz-1] # location of derivative for upper BC
 l0, l1, l2, l3 = weights2( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , zj )
 if upper_BC_flag == 'dirchlet':
   l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
 pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 return pzz


def partial_z( z , lower_BC_flag , upper_BC_flag ):
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

 # lower BC
 l0, l1, l2, l3 = weights( -z[0] , z[0] , z[1] , z[2] , z[0] )
 if lower_BC_flag == 'dirchlet':
   l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
 if lower_BC_flag == 'neumann':
   l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
 pz[0,0:3] = [ l1 , l2 , l3 ]
   
 # upper (far field) BC
 l0, l1, l2, l3 = weights( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , z[Nz-1] )
 if upper_BC_flag == 'dirchlet':
   l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
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


def fornberg_weights(z,x,m):
# From Bengt Fornbergs (1998) SIAM Review paper.
#  	Input Parameters
#	z location where approximations are to be accurate,
#	x(0:nd) grid point locations, found in x(0:n)
#	n one less than total number of grid points; n must
#	not exceed the parameter nd below,
#	nd dimension of x- and c-arrays in calling program
#	x(0:nd) and c(0:nd,0:m), respectively,
#	m highest derivative for which weights are sought,
#	Output Parameter
#	c(0:nd,0:m) weights at grid locations x(0:n) for derivatives
#	of order 0:m, found in c(0:n,0:m)
#      	dimension x(0:nd),c(0:nd,0:m)

  n = np.shape(x)[0]-1
  c = np.zeros([n+1,m+1])
  c1 = 1.0
  c4 = x[0]-z
  for k in range(0,m+1):  
    for j in range(0,n+1): 
      c[j,k] = 0.0
  c[0,0] = 1.0
  for i in range(0,n+1):
    mn = min(i,m)
    c2 = 1.0
    c5 = c4
    c4 = x[i]-z
    for j in range(0,i):
      c3 = x[i]-x[j]
      c2 = c2*c3
      if (j == i-1):
        for k in range(mn,0,-1): 
          c[i,k] = c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2
      c[i,0] = -c1*c5*c[i-1,0]/c2
      for k in range(mn,0,-1):
        c[j,k] = (c4*c[j,k]-k*c[j,k-1])/c3
      c[j,0] = c4*c[j,0]/c3
    c1 = c2
  return c


def diff_matrix( grid_params , lower_BC_flag , upper_BC_flag , diff_order , stencil_size ):
 # uses ghost nodes for dirchlet/neumann bcs.
 # make stencil size odd!
 # no interpolation

 Nz = grid_params['Nz']
 z = grid_params['z']
 H = grid_params['H']

 if stencil_size == 3:
   Dm1 = np.zeros([Nz-1])
   D0 = np.zeros([Nz])
   Dp1 = np.zeros([Nz-1])
   for j in range(1,Nz-1):
     Dm1[j-1],D0[j],Dp1[j] = fornberg_weights(z[j],z[j-1:j+2],diff_order)[:,diff_order]
   pzz = np.diag(Dp1,k=1) + np.diag(Dm1,k=-1) + np.diag(D0,k=0) 

   # lower (wall) BC sets variable to zero at the wall
   l0, l1, l2, l3 = fornberg_weights(z[0], np.append(-z[0],z[0:3]) ,diff_order)[:,diff_order]
   if lower_BC_flag == 'dirchlet':
     l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
   if lower_BC_flag == 'neumann':
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
   pzz[0,0:3] = [ l1 , l2 , l3 ]

   # upper (far field) BC
   l0, l1, l2, l3 = fornberg_weights(z[Nz-1], np.append(z[Nz-3:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
   if upper_BC_flag == 'dirchlet':
     l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
   if upper_BC_flag == 'neumann':
     l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
   pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]


 if stencil_size == 5:
   Dm2 = np.zeros([Nz-2])
   Dm1 = np.zeros([Nz-1])
   D0 = np.zeros([Nz])
   Dp1 = np.zeros([Nz-1])
   Dp2 = np.zeros([Nz-2])
   for j in range(2,Nz-2):
     Dm2[j-2],Dm1[j-1],D0[j],Dp1[j],Dp2[j] = fornberg_weights(z[j],z[j-2:j+3],diff_order)[:,diff_order]
   pzz = np.diag(Dp2,k=2) + np.diag(Dp1,k=1) + np.diag(D0,k=0) + np.diag(Dm1,k=-1) + np.diag(Dm2,k=-2) 
 

   # lower (wall) BC sets variable to zero at the wall
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[0], np.append(-z[0],z[0:5]) ,diff_order)[:,diff_order]
   if lower_BC_flag == 'dirchlet':
     l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
   if lower_BC_flag == 'neumann':
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
   pzz[0,0:5] = [ l1 , l2 , l3, l4 , l5 ]
   # lower (wall) BC sets variable to zero at the wall
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[1], np.append(-z[0],z[0:5]) ,diff_order)[:,diff_order]
   if lower_BC_flag == 'dirchlet':
     l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
   if lower_BC_flag == 'neumann':
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
   pzz[1,0:5] = [ l1 , l2 , l3, l4 , l5 ]
   # upper (far field) BC
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[Nz-1], np.append(z[Nz-5:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
   if upper_BC_flag == 'dirchlet':
     l4 = l4 - l5 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
   if upper_BC_flag == 'neumann':
     l4 = l4 + l5 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
   pzz[Nz-1,Nz-5:Nz] = [ l0 , l1 , l2 , l3 , l4 ]
   # upper (far field) BC
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[Nz-2], np.append(z[Nz-5:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
   if upper_BC_flag == 'dirchlet':
     l4 = l4 - l5 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
   if upper_BC_flag == 'neumann':
     l4 = l4 + l5 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
   pzz[Nz-2,Nz-5:Nz] = [ l0 , l1 , l2 , l3 , l4 ]


 return pzz


# =============================================================================
# loop over Nz resolution Chebyshev node grid


max_exp = 12 # power of two, must be equal to or greater than 5 (maximum N = 2^max)
Nr = np.power(np.ones([max_exp-3])*2.,np.linspace(4.,max_exp,max_exp-3)) # resolution 
#print(Nr)
Ng = int(np.shape(Nr)[0]) # number of resolutions to try

Linf1 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid

Linf2 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 
Linf2f = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2cf = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 

Linf12 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c2 = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid

Linf22 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c2 = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 

Linf2f = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2cf = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 
Linf22f = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c2f = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 

Linf4 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf4c = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid

Linf42 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf4c2 = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid

H = 1.0 # domain height

for n in range(0,Ng): 
  
  Nz = int(Nr[n]) # resolution
  print('Number of grid points: ',Nz)
 
  dz = H/Nz
  z = np.linspace(dz, Nz*dz, num=Nz)-dz/2. # uniform grid

  zc = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H
  zc = zc[0:Nz] # half cosine grid
  dzc = zc[1:Nz] - zc[0:Nz-1]

  grid_params = { 'z':z, 'dz':dz, 'Nz':Nz , 'H':H}
  grid_params2 = { 'z':zc, 'dz':dzc, 'Nz':Nz , 'H':H}
  
  U0 = 2. # free stream velocity
  m = np.pi/(2.*H)
  q = 2.*np.pi/(H)

  u = np.zeros([Nz,1]); uz = np.zeros([Nz,1]); uzz = np.zeros([Nz,1]); 
  uzzz = np.zeros([Nz,1]); uzzzz = np.zeros([Nz,1]);
  u[:,0] = U0*np.sin(m*z); uz[:,0] = U0*m*np.cos(m*z) # du/dz
  uzz[:,0] = -U0*m**2.*np.sin(m*z) # d^2u/dz^2
  uzzz[:,0] = -U0*m**3.*np.cos(m*z) # d^3u/dz^3
  uzzzz[:,0] = U0*m**4.*np.sin(m*z) # d^3u/dz^3

  uc = np.zeros([Nz,1]); uzc = np.zeros([Nz,1]); uzzc = np.zeros([Nz,1]); 
  uzzzc = np.zeros([Nz,1]); uzzzzc = np.zeros([Nz,1]);
  uc[:,0] = U0*np.sin(m*zc); uzc[:,0] = U0*m*np.cos(m*zc) 
  uzzc[:,0] = -U0*m**2.*np.sin(m*zc)
  uzzzc[:,0] = -U0*m**3.*np.cos(m*zc) # d^3u/dz^3
  uzzzzc[:,0] = U0*m**4.*np.sin(m*zc) # d^3u/dz^3
  
  b = np.zeros([Nz,1]); bz = np.zeros([Nz,1]); bzz = np.zeros([Nz,1]); 
  bzzz = np.zeros([Nz,1]); bzzzz = np.zeros([Nz,1])
  b[:,0] = U0*np.cos(q*z); bz[:,0] = -U0*q*np.sin(q*z) 
  bzz[:,0] = -U0*q**2.*np.cos(q*z) 
  bzzz[:,0] = U0*q**3.*np.sin(q*z) 
  bzzzz[:,0] = U0*q**4.*np.cos(q*z) 

  bc = np.zeros([Nz,1]); bzc = np.zeros([Nz,1]); bzzc = np.zeros([Nz,1]); 
  bzzzc = np.zeros([Nz,1]); bzzzzc = np.zeros([Nz,1])
  bc[:,0] = U0*np.cos(q*zc); bzc[:,0] = -U0*q*np.sin(q*zc) 
  bzzc[:,0] = -U0*q**2.*np.cos(q*zc)
  bzzzc[:,0] = U0*q**3.*np.sin(q*zc) 
  bzzzzc[:,0] = U0*q**4.*np.cos(q*zc) 


  # try 2nd derivative, both grids, stencil 3 and 5
  #D=diff_matrix( grid_params2 , 'dirchlet' , 'neumann' , diff_order=2 , stencil_size=3 )
  #D2 = partial_zz( zc , 'dirchlet' , 'neumann' )

  # compute 1st derivatives:
  uz0 = np.dot( partial_z( z , 'dirchlet' , 'neumann' ) , u ) # uniform grid  
  uz0c = np.dot( partial_z( zc , 'dirchlet' , 'neumann' ) , uc ) # cosine grid

  # compute 2nd derivatives:
  uzz0 = np.dot( partial_zz( z , 'dirchlet' , 'neumann' ) , u ) # uniform grid
  uzz0c = np.dot( partial_zz( zc , 'dirchlet' , 'neumann' ) , uc ) # cosine grid
  uzz0f = np.dot( diff_matrix( grid_params , 'dirchlet' , 'neumann' , diff_order=2 , stencil_size=3 ) , u ) # uniform grid
  uzz0cf = np.dot( diff_matrix( grid_params2 , 'dirchlet' , 'neumann' , diff_order=2 , stencil_size=3 ) , uc ) # cosine grid

  # compute 4th derivatives:
  uzzzzf = np.dot( diff_matrix( grid_params , 'dirchlet' , 'neumann' , diff_order=4 , stencil_size=5 ) , u ) # uniform grid
  uzzzzcf = np.dot( diff_matrix( grid_params2 , 'dirchlet' , 'neumann' , diff_order=4 , stencil_size=5 ) , uc ) # cosine grid

  # compute 4th derivatives:
  bzzzzf = np.dot( diff_matrix( grid_params , 'neumann' , 'neumann' , diff_order=4 , stencil_size=5 ) , b ) # uniform grid
  bzzzzcf = np.dot( diff_matrix( grid_params2 , 'neumann' , 'neumann' , diff_order=4 , stencil_size=5 ) , bc ) # cosine grid

  # compute 1st derivatives:
  bz0 = np.dot( partial_z( z , 'neumann' , 'neumann' ) , b ) # uniform grid  
  bz0c = np.dot( partial_z( zc , 'neumann' , 'neumann' ) , bc ) # cosine grid

  # compute 2nd dericatives:
  bzz0 = np.dot( partial_zz( z , 'neumann' , 'neumann' ) , b ) # uniform grid
  bzz0c = np.dot( partial_zz( zc , 'neumann' , 'neumann' ) , bc ) # cosine grid
  bzz0f = np.dot( diff_matrix( grid_params , 'neumann' , 'neumann' , diff_order=2 , stencil_size=3 ) , b ) # uniform grid
  bzz0cf = np.dot( diff_matrix( grid_params2 , 'neumann' , 'neumann' , diff_order=2 , stencil_size=3 ) , bc ) # cosine grid

  Linf1[n] = np.amax(abs(uz-uz0)/abs(m*U0)) 
  Linf1c[n] = np.amax(abs(uzc-uz0c)/abs(m*U0))

  Linf2[n] = np.amax(abs(uzz-uzz0)/abs(m**2.*U0)) 
  Linf2c[n] = np.amax(abs(uzzc-uzz0c)/abs(m**2.*U0))
  Linf2f[n] = np.amax(abs(uzz-uzz0f)/abs(m**2.*U0)) 
  Linf2cf[n] = np.amax(abs(uzzc-uzz0cf)/abs(m**2.*U0))

  Linf4[n] = np.amax(abs(uzzzz-uzzzzf)/abs(m**4.*U0)) 
  Linf4c[n] = np.amax(abs(uzzzzc-uzzzzcf)/abs(m**4.*U0))

  Linf42[n] = np.amax(abs(bzzzz-bzzzzf)/abs(q**4.*U0)) 
  Linf4c2[n] = np.amax(abs(bzzzzc-bzzzzcf)/abs(q**4.*U0))

  Linf12[n] = np.amax(abs(bz-bz0)/abs(q*U0)) 
  Linf1c2[n] = np.amax(abs(bzc-bz0c)/abs(q*U0))

  Linf22[n] = np.amax(abs(bzz-bzz0)/abs(q**2.*U0)) 
  Linf2c2[n] = np.amax(abs(bzzc-bzz0c)/abs(q**2.*U0))
  Linf22f[n] = np.amax(abs(bzz-bzz0f)/abs(q**2.*U0)) 
  Linf2c2f[n] = np.amax(abs(bzzc-bzz0cf)/abs(q**2.*U0))


  if n == 7:
   plotname = figure_path + 'computed_4th_derivative_uniform_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(-uzzzz/(m**4.*U0),z,'k',label=r"analytical")
   plt.plot(-uzzzzf/(m**4.*U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_4th_derivative_cosine_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(-uzzzzc/(m**4.*U0),zc,'k',label=r"analytical")
   plt.plot(-uzzzzcf/(m**4.*U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_4th_derivative_uniform_grid_error.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uzz/(m*U0)-uzz0/(m*U0)),abs(uzz/(m*U0))),z,'k') 
   plt.plot(abs(uzzzz-uzzzzf)/abs(m**4.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_4th_derivative_cosine_grid_error.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uzzc/(m*U0)-uzz0c/(m*U0)),abs(uzzc/(m*U0))),z,'k') 
   plt.plot(abs(uzzzzc-uzzzzcf)/abs(m**4.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_4th_derivative_uniform_grid2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(-bzzzz/(q**4.*U0),z,'k',label=r"analytical")
   plt.plot(-bzzzzf/(q**4.*U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_4th_derivative_cosine_grid2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(-bzzzzc/(q**4.*U0),zc,'k',label=r"analytical")
   plt.plot(-bzzzzcf/(q**4.*U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_4th_derivative_uniform_grid_error2.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uzz/(m*U0)-uzz0/(m*U0)),abs(uzz/(m*U0))),z,'k') 
   plt.plot(abs(bzzzz-bzzzzf)/abs(q**4.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_4th_derivative_cosine_grid_error2.png'
   fig = plt.figure(figsize=(8,8))
   #plt.plot(np.divide(abs(uzzc/(m*U0)-uzz0c/(m*U0)),abs(uzzc/(m*U0))),z,'k') 
   plt.plot(abs(bzzzzc-bzzzzcf)/abs(q**4.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 4$^{th}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

  """
  D=diff_matrix( grid_params , 'neumann' , 'neumann' , diff_order=4 , stencil_size=5 )
  print(D[0,:])
  print(D[1,:])
  #print(D[2,:])
  #print(D[13,:])
  print()
  print(D[14,:])
  print(D[15,:])
  """
  """
  print(D[0,:])
  print(D2[0,:])
  print()
  print(D[1,:])
  print(D2[1,:])
  print()
  print(D[14,:])
  print(D2[14,:])
  print()
  print(D[15,:])
  print(D2[15,:])
  """
  #uzz0 = np.dot( partial_zz( z , 'dirchlet' , 'neumann' ) , u ) # uniform grid
  #uzz0c = np.dot( partial_zz( zc , 'dirchlet' , 'neumann' ) , uc ) # cosine grid



plotname = figure_path + 'first_derivative_error.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf1,'r',label=r"uniform")
plt.loglog(Nr,Linf1c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1$^{st}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'fourth_derivative_error.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf4,'r',label=r"uniform")
plt.loglog(Nr,Linf4c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"4$^{th}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + 'fourth_derivative_error2.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf42,'r',label=r"uniform")
plt.loglog(Nr,Linf4c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"4$^{th}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'second_derivative_error.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf2,'r',label=r"uniform")
plt.loglog(Nr,Linf2c,'b',label=r"cosine")
plt.loglog(Nr,Linf2f,'--g',label=r"uniform")
plt.loglog(Nr,Linf2cf,'--k',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2$^{nd}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + 'first_derivative_error_case2.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1$^{st}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'second_derivative_error_case2.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf22,'r',label=r"uniform")
plt.loglog(Nr,Linf2c2,'b',label=r"cosine")
plt.loglog(Nr,Linf22f,'--g',label=r"uniform")
plt.loglog(Nr,Linf2c2f,'--k',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2$^{nd}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


"""
max_exp = 15 # power of two, must be equal to or greater than 5 (maximum N = 2^max)
Nr = np.power(np.ones([max_exp-3])*2.,np.linspace(4.,max_exp,max_exp-3)) # resolution 
Ng = int(np.shape(Nr)[0]) # number of resolutions to try

Linf1 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid

Linf2 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 

Linf12 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c2 = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid

Linf22 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c2 = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 
 
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

  q = 2.*np.pi/(H)
  b = np.zeros([Nz,1])
  bz = np.zeros([Nz,1])
  bzz = np.zeros([Nz,1])
  b[:,0] = U0*np.cos(q*z)
  bz[:,0] = -U0*q*np.sin(q*z) 
  bzz[:,0] = -U0*q**2.*np.cos(q*z) 

  bc = np.zeros([Nz,1])
  bzc = np.zeros([Nz,1])
  bzzc = np.zeros([Nz,1])
  bc[:,0] = U0*np.cos(q*zc) 
  bzc[:,0] = -U0*q*np.sin(q*zc) 
  bzzc[:,0] = -U0*q**2.*np.cos(q*zc)

  if n == 6:
   plotname = figure_path + 'solutions_uniform_grid_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(b/U0,z,'b',label=r"$b$")
   plt.plot(-bz/(q*U0),z,'k',label=r"$b_z$")
   plt.plot(-bzz/(q**2.*U0),z,'--r',label=r"$b_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=3,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'solutions_cosine_grid_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(bc/U0,zc,'b',label=r"$b$")
   plt.plot(-bzc/(q*U0),zc,'k',label=r"$b_z$")
   plt.plot(-bzzc/(q**2.*U0),zc,'--r',label=r"$b_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=3,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

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

  # compute 1st derivatives:
  uz0 = np.dot( partial_z( z , 'dirchlet' , 'neumann' ) , u ) # uniform grid  
  uz0c = np.dot( partial_z( zc , 'dirchlet' , 'neumann' ) , uc ) # cosine grid

  # compute 2nd dericatives:
  uzz0 = np.dot( partial_zz( z , 'dirchlet' , 'neumann' ) , u ) # uniform grid
  uzz0c = np.dot( partial_zz( zc , 'dirchlet' , 'neumann' ) , uc ) # cosine grid

  # compute 1st derivatives:
  bz0 = np.dot( partial_z( z , 'neumann' , 'neumann' ) , b ) # uniform grid  
  bz0c = np.dot( partial_z( zc , 'neumann' , 'neumann' ) , bc ) # cosine grid

  # compute 2nd dericatives:
  bzz0 = np.dot( partial_zz( z , 'neumann' , 'neumann' ) , b ) # uniform grid
  bzz0c = np.dot( partial_zz( zc , 'neumann' , 'neumann' ) , bc ) # cosine grid

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


   plotname = figure_path + 'computed_1st_derivative_uniform_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bz-bz0)/abs(q*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_cosine_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bzc-bz0c)/abs(q*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_uniform_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bzz-bzz0)/abs(q**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bzzc-bzz0c)/abs(q**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);


  Linf1[n] = np.amax(abs(uz-uz0)/abs(m*U0)) 
  Linf1c[n] = np.amax(abs(uzc-uz0c)/abs(m*U0))

  Linf2[n] = np.amax(abs(uzz-uzz0)/abs(m**2.*U0)) 
  Linf2c[n] = np.amax(abs(uzzc-uzz0c)/abs(m**2.*U0))

  Linf12[n] = np.amax(abs(bz-bz0)/abs(q*U0)) 
  Linf1c2[n] = np.amax(abs(bzc-bz0c)/abs(q*U0))

  Linf22[n] = np.amax(abs(bzz-bzz0)/abs(q**2.*U0)) 
  Linf2c2[n] = np.amax(abs(bzzc-bzz0c)/abs(q**2.*U0))


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


plotname = figure_path + 'first_derivative_error_case2.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1$^{st}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'second_derivative_error_case2.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf22,'r',label=r"uniform")
plt.loglog(Nr,Linf2c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2$^{nd}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

"""
