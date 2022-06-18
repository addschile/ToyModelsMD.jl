include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using DelimitedFiles
using MPI

function cbfunc(xstore::Array,system::AbstractSystem,args)
  xstore[args[2][1],:] .= copy(system.x)
end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

ntraj = 10
ntrajperproc = (ntraj/nprocs)
dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

# define system
mb = MullerBrown()

#MPI.Barrier(comm)
#MPI.Bcast(mb,rank,comm)

for i in 1:ntrajperproc
  seed::Int64 = rank*ntrajperproc + i
  # make system and thermostat
  langevin = Langevin(1.0,1.0,convert(UInt64,seed))
  system = ThermostattedSystem(mb,langevin)
  # define callback function
  xstore = zeros(nsteps,2)
  cbf(system,args...) = cbfunc(xstore,system,args)
  cb = Callback(1,cbf)
  # set initial condition and run
  system.x = [0.63, 0.03]
  integrator = StochasticEuler(system,mb)
  runtraj!(nsteps,dt,integrator,cb)
  writedlm("parallel_traj_$seed.txt",xstore)
  println("Traj $seed")
  println(xstore)
  println("")
end
