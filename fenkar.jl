module Fenkar

using StaticArrays

export fenkar!, fenkar

# define fenkar system
function H(x;k=100.0)
	return 0.5*(1.0 + tanh(k*x))
end
function fenkar!(dx, x, p, t) # in-place, no SA, fast
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi = p
	
	# fenkar dynamics
	dx[1] = -((x[1]/to)*H(uc-x[1]) + H(x[1]-uc)/tr - (x[2]/td)*H(x[1]-uc)*(1.0-x[1])*(x[1]-uc) - (x[3]/tsi)*H(x[1]-ucsi;k=xk))
	dx[2] = H(uc-x[1])*(1.0-x[2])/(tv1m*H(uv-x[1]) + tv2m*H(x[1]-uv)) - H(x[1]-uc)*x[2]/tvp
	dx[3] = H(uc-x[1])*(1.0-x[3])/twm - H(x[1]-uc)*x[3]/twp
	
	return nothing
end
function fenkar(x,p,t)		# out-of-place, SA, faster
	
	# parameters
	@views tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi = p
	
	# fenkar dynamics
	du = -((x[1]/to)*H(uc-x[1]) + H(x[1]-uc)/tr - (x[2]/td)*H(x[1]-uc)*(1.0-x[1])*(x[1]-uc) - (x[3]/tsi)*H(x[1]-ucsi;k=xk))
	dv = H(uc-x[1])*(1.0-x[2])/(tv1m*H(uv-x[1]) + tv2m*H(x[1]-uv)) - H(x[1]-uc)*x[2]/tvp
	dw = H(uc-x[1])*(1.0-x[3])/twm - H(x[1]-uc)*x[3]/twp
	
	return SA[du,dv,dw]
end


# fenkar system parameters in p[1:13] (BR; from https://doi.org/10.1063/1.166311) 
#         tsi,  tv1m,   tv2m,  tvp,  twm,    twp,   td,   to,   tr,   xk,  uc,     uv,  ucsi
#   p = [29.0,  19.6, 1250.0, 3.33, 41.0,  870.0, 0.25, 12.5, 33.3, 10.0, 0.13,  0.04,  0.85]
   p = SA[ 22.0, 333.0,   40.0, 10.0, 65.0, 1000.0, 0.12, 12.5, 25.0, 10.0, 0.13, 0.025,  0.85]
   u0= SA[0.2, 0.5, 0.8]
   tspan = (0.0,1000.0)
end

