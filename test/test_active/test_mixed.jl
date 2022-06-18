include("../../src/ToyModelsMD.jl")
using .ToyModelsMD

function cbfunc(xstore::Array,system::AbstractSystem,args...)
  # calculate the stuff
  push!(xstore,copy(system.x))
end

### define system ###
# mueller-brown potential
mb = MullerBrown()
# gaussian force model
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
# mixed model
model = MixedModel(2,[mb,gm])
# thermostat and system
alangevin = ActiveLangevin(2,1.0,1.0,1.0,1.0)
system = ActiveBrownianSystem(3,0.0,0.0,[0.63,0.03,0.0],zeros(Float64,2),alangevin)

# define callback function
xstore = []
cbf(system,args...) = cbfunc(xstore,system,args)
cb = Callback(1,cbf)

dt = 0.0001
tf = 0.0015
nsteps = convert(Int64, round(tf/dt))

system.x = [0.63, 0.03,0.0]
integrator = StochasticEuler(system,model)
runtraj!(nsteps,dt,integrator,cb)
println(xstore)
