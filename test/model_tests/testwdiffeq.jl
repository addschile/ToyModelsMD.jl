using DifferentialEquations
using Plots
include("../src/ToyModelsMD.jl")
using .ToyModelsMD

mb = MullerBrown()
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(mb,langevin)

#fd(du,u,p,t) = fderivative(du,u,p,t;sys=system)
#gd(du,u,p,t) = gderivative(du,u,p,t;therm=langevin)

fd(du,u,p,t) = fderivative(du,u,system)
gd(du,u,p,t) = gderivative(du,u,langevin)

u0 = [0.63,0.03]
prob = SDEProblem(fd,gd,u0,(0.0,0.0015))
sol = solve(prob,EM(),dt=0.0001)
println(sol.u)
#plt = plot(sol)
#display(plt)
