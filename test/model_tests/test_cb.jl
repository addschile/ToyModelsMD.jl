include("../../src/ToyModelsMD.jl")
using .ToyModelsMD

function cbfunc(xstore::Array,system::AbstractSystem,args...)
  # calculate the stuff
  push!(xstore,copy(system.x))
end

# define system
mb = MullerBrown()
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(mb,langevin)

# define callback function
xstore = []
cbf(system,args...) = cbfunc(xstore,system,args)
cb = Callback(1,cbf)

dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

system.x = [0.63, 0.03]
integrator = StochasticEuler(system,mb)
runtraj!(nsteps,dt,integrator,cb)
println(xstore)
