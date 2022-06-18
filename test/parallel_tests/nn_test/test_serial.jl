include("../../../src/ToyModelsMD.jl")
using .ToyModelsMD
using Flux
using DelimitedFiles

function cbfunc(xstore::Array,rstore::Array,fstore::Array,system::AbstractSystem,args)
  model = args[1]
  fstore[args[2][1],1,:] .= copy(model.potentials[1].f)
  fstore[args[2][1],2,:] .= copy(model.potentials[2].f)
  xstore[args[2][1],:] .= copy(system.x)
  rstore[args[2][1],:] .= copy(system.thermostat.rands)
end

ntraj = 10
dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

### define system ###
# mueller-brown potential
mb = MullerBrown()
# neural network force model
model = Chain(Dense(3,10),Dense(10,2))
theta,re = Flux.destructure(model)
theta .= 0.1
model = re(theta)
nn = NNModel(2,model)
# mixed model
mmodel = MixedModel(2,[mb,nn])

# define callback function
xstore = zeros(nsteps,2)
rstore = zeros(nsteps,2)
fstore = zeros(nsteps,2,2)
cbf(system,args...) = cbfunc(xstore,rstore,fstore,system,args)
cb = Callback(1,cbf)

for i in 1:ntraj
  # thermostat and system
  langevin = Langevin(1.0,1.0,convert(UInt64,i))
  system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)
  xstore .= 0.0
  rstore .= 0.0
  fstore .= 0.0
  # set initial condition and run
  integrator = StochasticEuler(system,mmodel)
  runtraj!(nsteps,dt,integrator,cb)
  writedlm("serial_traj_$i.txt",xstore)
  writedlm("serial_rands_$i.txt",rstore)
  writedlm("serial_forces_1_$i.txt",fstore[:,1,:])
  writedlm("serial_forces_2_$i.txt",fstore[:,2,:])
end 
