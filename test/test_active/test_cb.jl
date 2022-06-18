include("../../src/ToyModelsMD.jl")
using .ToyModelsMD

function cbfunc(xstore::Array,system::AbstractSystem,args...)
  # calculate the stuff
  push!(xstore,copy(system.x))
end

# define system
mb = MullerBrown()
alangevin = ActiveLangevin(2,1.0,1.0,1.0,1.0)
system = ActiveBrownianSystem(1.0,mb,alangevin)

# define callback function
xstore = []
cbf(system,args...) = cbfunc(xstore,system,args)
cb = Callback(1,cbf)

dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

system.x = [0.63, 0.03, 0.0]
integrator = StochasticEuler(system,mb)
runtraj!(nsteps,dt,integrator,cb)
println(xstore)
