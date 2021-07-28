using DifferentialEquations
using Plots
include("../src/potentials.jl")
include("../src/thermostats.jl")
include("../src/systems.jl")

mb = MullerBrown()
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(mb,langevin)

#fd(du,u,p,t) = fderivative(du,u,p,t;sys=system)
#gd(du,u,p,t) = gderivative(du,u,p,t;therm=langevin)

fd(du,u,p,t) = fderivative(du,u,system)
gd(du,u,p,t) = gderivative(du,u,langevin)

u0 = [0.63,0.03]
prob = SDEProblem(fd,gd,u0,(0.0,0.0015))
#prob = SDEProblem(fderivative,gderivative,u0,(0.0,0.15))
sol = solve(prob,EM(),dt=0.0001)
println(sol)
plotly()
plot(sol)
