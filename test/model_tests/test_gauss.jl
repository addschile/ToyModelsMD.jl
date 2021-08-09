include("../../src/ToyModelsMD.jl")
using .ToyModelsMD

function cbfunc(xstore::Array,rstore::Array,system::AbstractSystem,args...)
  # calculate the stuff
  push!(xstore,copy(system.x))
  push!(rstore,copy(system.thermostat.rands))
end

# define system
Mx = 21
My = 21
Mt = 21
shape = (Mx*My*Mt,2)
mux = rand(Float64,21)
muy = rand(Float64,21)
mut = rand(Float64,21)
sigx = rand(Float64,21)
sigy = rand(Float64,21)
sigt = rand(Float64,21)
cs = zeros(Float64,shape)
gm = GaussianModel2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,cs)
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(gm,langevin)

# define callback function
xstore = []
rstore = []
cbf(system,args...) = cbfunc(xstore,rstore,system,args)
cb = Callback(1,cbf)

dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

system.x = [0.63, 0.03]
integrator = StochasticEuler(system,gm)
runtraj!(nsteps,dt,integrator,cb)
println(xstore)
println(dt*rstore)
