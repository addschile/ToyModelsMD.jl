include("../../src/ToyModelsMD.jl")
using .ToyModelsMD

mb = MullerBrown()
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(mb,langevin)

dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

system.x = [0.63, 0.03]
integrator = StochasticEuler(system,mb)
runtraj!(nsteps,dt,integrator)
