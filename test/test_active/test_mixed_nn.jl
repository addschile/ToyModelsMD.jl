include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using Flux

function cbfunc(xstore::Array,system::AbstractSystem,args...)
  # calculate the stuff
  push!(xstore,copy(system.x))
end

### define system ###
# mueller-brown potential
mb = MullerBrown()
# neural network force model
nn = NNModel(3,Chain(Dense(4,10),Dense(10,3)))
# mixed model
model = MixedModel(3,[mb,nn])
# thermostat and system
alangevin = ActiveLangevin(3,1.0,1.0,1.0,1.0)
system = ActiveBrownianSystem(3,0.0,0.0,[0.63,0.03,0.0],zeros(Float64,2),alangevin)

# define callback function
xstore = []
cbf(system,args...) = cbfunc(xstore,system,args)
cb = Callback(1,cbf)

dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

system.x = [0.63, 0.03, 0.0]
integrator = StochasticEuler(system,model)
runtraj!(nsteps,dt,integrator,cb)
println(xstore)
