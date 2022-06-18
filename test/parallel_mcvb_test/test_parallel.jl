using MPI
include("../../../src/ToyModelsMD.jl")
using .ToyModelsMD
using Flux
using DelimitedFiles

function cbfunc(xstore::Array,rstore::Array,system::AbstractSystem,args)
  xstore[args[2][1],:] .= copy(system.x)
  rstore[args[2][1],:] .= copy(system.thermostat.rands)
end

#function cbfunc(xstore::Array,system::AbstractSystem,args)
#  xstore[args[2][1],:] .= copy(system.x)
#end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

ntraj = 10
ntrajperproc = (ntraj/nprocs)
dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

### define system ###
# mueller-brown potential
mb = MullerBrown()
# neural network force model
model = Chain(Dense(3,10),Dense(10,2))
theta,re = Flux.destructure(model)
if rank == 0
  theta .= 0.1
end
MPI.Barrier(comm)
MPI.Bcast!(theta,0,comm)
model = re(theta)
nn = NNModel(2,model)
# mixed model
mmodel = MixedModel(2,[mb,nn])

# define callback function
xstore = zeros(nsteps,2)
rstore = zeros(nsteps,2)
cbf(system,args...) = cbfunc(xstore,rstore,system,args)
cb = Callback(1,cbf)

for i in 1:ntrajperproc
  seed::Int64 = rank*ntrajperproc + i
  # make system and thermostat
  langevin = Langevin(1.0,1.0,convert(UInt64,seed))
  system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)
  # set initial condition and run
  xstore .= 0.0
  rstore .= 0.0
  integrator = StochasticEuler(system,mmodel)
  runtraj!(nsteps,dt,integrator,cb)
  #writedlm("parallel_traj_$seed.txt",xstore)
  writedlm("new_parallel_traj_$seed.txt",xstore)
  writedlm("parallel_rands_$seed.txt",rstore)
  #println("Traj $seed")
  #println(xstore)
  #println("")
end

#for i in 1:ntraj
#  # thermostat and system
#  langevin = Langevin(1.0,1.0,convert(UInt64,i))
#  system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)
#  # define callback function
#  xstore = zeros(nsteps,2)
#  cbf(system,args...) = cbfunc(xstore,system,args)
#  cb = Callback(1,cbf)
#  # set initial condition and run
#  system.x = [0.63, 0.03]
#  integrator = StochasticEuler(system,mb)
#  runtraj!(nsteps,dt,integrator,cb)
#  writedlm("serial_traj_$i.txt",xstore)
#  println("Traj $i")
#  println(xstore)
#  println("")
#end 
