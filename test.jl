include("fenkar.jl") 
using StaticArrays, DifferentialEquations, ModelingToolkit, Plots

#= 	fenkar.jl has an in-place FenKar function (fenkar!), but it is slower than the 
	out-of-place version using StaticArrays (fenkar), go figure.
		Basic version is  ~33712x faster than real time (Tsit5())
		SArray version is ~48685x faster than real time (Tsit5())
	So we use the SArray one (fenkar), unless managing static arrays gets annoying, 
	which they often do, then just use fenkar!.
=#

tspan = Fenkar.tspan
u0 = rand(Float64,3)
p  = [ 22.0, 333.0,   40.0, 10.0, 65.0, 1000.0, 0.12, 12.5, 25.0, 10.0, 0.13, 0.025,  0.85]

prob = ODEProblem(Fenkar.fenkar!, u0, tspan, p)

sol = solve(prob, Tsit5())
plot(sol)

using OrdinaryDiffEq, ForwardDiff, MinimallyDisruptiveCurves, Statistics, Plots, LinearAlgebra, LaTeXStrings

nom_prob = prob
nom_sol = sol
t = collect(range(tspan[1], tspan[2]; length=1025))

## Model features of interest are mean prey population, and max predator population (over time)
function features(p)
  prob = remake(nom_prob; p=p)
  sol  = solve(prob, Tsit5(); saveat = t)
  return [maximum(sol[1,:]), sum(abs.(sol[1,:])), sum(sol[1,:].^2)]
  #return sol[1,1:2:100]
end

nom_features = features(p)

## loss function, we can take as l2 difference of features vs nominal features
function loss(p)
  prob = remake(nom_prob; p=p)
  p_features = features(p)
  loss = sum(abs2, p_features - nom_features)
  return loss
end

## gradient of loss function
function lossgrad(p,g)
  g[:] = ForwardDiff.gradient(p) do p
    loss(p)
  end
  return loss(p)
end

## package the loss and gradient into a DiffCost structure
cost = DiffCost(loss, lossgrad)

#=
We evaluate the hessian once only, at p.
Why? to find locally insensitive directions of parameter perturbation
The small eigenvalues of the Hessian are one easy way of defining these directions 
=#
#which md curve to plot
#which_dir = 2
hess0 = ForwardDiff.hessian(loss,p)
ev(i) = eigen(hess0).vectors[:,i]
which_dir = 6 #argmin(abs.(eigvals(hess0)))

## Now we set up a minimally disruptive curve, with nominal parameters p and initial direction ev(1) 
init_dir = ev(which_dir); momentum = 1.; span = (-100.,100.)
curve_prob = MDCProblem(cost, p, init_dir, momentum, span)
@time mdc = evolve(curve_prob, Tsit5)

function sol_at_p(p)
  prob = remake(nom_prob; p=p)
  sol = solve(prob, Tsit5())
end

cost_vec = [mdc.cost(el) for el in eachcol(trajectory(mdc))]

#tsi,tv1m,tv2m,tvp,twm,twp,td,to,tr,xk,uc,uv,ucsi
p1 = plot(mdc; pnames=["tsi","tv1m","tv2m","tvp","twm","twp","td","to","tr","xk","uc","uv","ucsi"])
p2 = plot(distances(mdc), log.(cost_vec), ylabel = "log(cost)", xlabel = "distance", title = "cost over MD curve");
mdc_plot = plot(p1,p2, layout=(2,1), size = (800,800))

nominal_trajectory = plot(sol_at_p(mdc(0.)[:states]), label = ["u" "v" "w"])
perturbed_trajectory = plot(sol_at_p(mdc(75.)[:states]), label = ["u" "v" "w"])
traj_comparison = plot(nominal_trajectory, perturbed_trajectory, layout = (2,1), size = (800,800), xlabel = "time", ylabel = "state variable")

plot(mdc_plot, traj_comparison, layout=(1,2), size=(1200,900))
savefig("./fig.pdf")
