include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using Flux
using DelimitedFiles
using Plots

function Afunc(system::AbstractSystem,mm::MixedModel,ind::Int64,dt::Float64)
  lam::Float64 = 2.0e3
  pot::Float64 = potential(system,mm.potentials[1])
  ind==1500 && system.x[2]>0.7 && pot<-145.0 ? lam : 0.0
end

### define system ###
dt = 0.0001
steps = 1500
tf = steps*dt

# mueller-brown potential
mb = MullerBrown()
# neural network model
nn = NNModel(2,Chain(Dense(3,10),Dense(10,2)))

# mixed model
model = MixedModel(2,[mb,nn])

# thermostat and system
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

# define value baseline function
vbl = NNValueBaseline(Chain(Dense(3,10),Dense(10,1)))

# define callback function
mcvb = MCVBCallback(Afunc,nn,vbl)
# set up integrator and run!
integrator = StochasticEuler(system,model)
runtraj!(steps,dt,integrator,mcvb)
println(mcvb.dkl)
writedlm("gradf.txt",mcvb.gradf)
writedlm("gradv.txt",mcvb.gradv*dt)
