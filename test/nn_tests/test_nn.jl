include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using Flux

function cbfunc(xstore::Array,system::AbstractSystem,args...)
  # calculate the stuff
  push!(xstore,copy(system.x))
end

### define system ###
nn = Chain(Dense(3,10),Dense(10,2))
model = NNModel(2,nn)
# thermostat and system
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

# define callback function
xstore = []
cbf(system,args...) = cbfunc(xstore,system,args)
cb = Callback(1,cbf)

dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

system.x = [0.63, 0.03]
integrator = StochasticEuler(system,model)
runtraj!(nsteps,dt,integrator,cb)
println(xstore)
