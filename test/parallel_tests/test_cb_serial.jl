include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using DelimitedFiles

function cbfunc(xstore::Array,system::AbstractSystem,args)
  xstore[args[2][1],:] .= copy(system.x)
end

ntraj = 10
dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

# define system
mb = MullerBrown()

for i in 1:ntraj
  # make system and thermostat
  langevin = Langevin(1.0,1.0,convert(UInt64,i))
  system = ThermostattedSystem(mb,langevin)
  # define callback function
  xstore = zeros(nsteps,2)
  cbf(system,args...) = cbfunc(xstore,system,args)
  cb = Callback(1,cbf)
  # set initial condition and run
  system.x = [0.63, 0.03]
  integrator = StochasticEuler(system,mb)
  runtraj!(nsteps,dt,integrator,cb)
  writedlm("serial_traj_$i.txt",xstore)
  println("Traj $i")
  println(xstore)
  println("")
end
