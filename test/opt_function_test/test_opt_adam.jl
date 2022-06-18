include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using Flux
using DelimitedFiles

function softAfunc(system::AbstractSystem,mm::MixedModel,ind::Int64,dt::Float64)
  lam::Float64 = 1.0e4/dt
  xf::Float64 = -0.5
  yf::Float64 = 1.5
  ind>=1500 ? -lam*((system.x[1]-xf)^2 + (system.x[2]-yf)^2) : 0.0
end

function Afunc(system::AbstractSystem,mm::MixedModel,ind::Int64,dt::Float64)
  lam::Float64 = 2.0e3
  pot::Float64 = potential(system,mm.potentials[1])
  ind>=1500 && system.x[2]>0.7 && pot<-145.0 ? lam : 0.0
end

function cbfunc(xstore::Array,system::AbstractSystem,args::Tuple{MixedModel, Tuple{Int64, Float64}})
  xstore[args[2][1],:] .= copy(system.x)
end

### define system ###
dt = 0.0001
nsteps = 1500
tf = nsteps*dt

# mueller-brown potential
mb = MullerBrown()

# neural network model
nntmp = Chain(Dense(3,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,2))
theta,re = Flux.destructure(nntmp)
theta .= 0.0
nntmp = re(theta)
nn = NNModel(2,nntmp)

# mixed model
model = MixedModel(2,[mb,nn])

# thermostat and system
langevin = Langevin(1.0,1.0,convert(UInt64,42))
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

# define value baseline function
nntmp = Chain(Dense(3,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,1))
theta,re = Flux.destructure(nntmp)
theta .= 0.0
nntmp = re(theta)
vbl = NNValueBaseline(nntmp)

# set up integrator
integrator = StochasticEuler(system,model)

# set up optimization loop with soft boundary conditions
softmcvb = MCVBCallback(softAfunc,nn,vbl)
nepochs = 20
ntraj = 5
lrf = 1.0#dt*50
lrv = 1.0#dt*50
beta1f = 0.9
beta2f = 0.999
beta1v = 0.9
beta2v = 0.999
eps = 1.0e-8
mcvbtrainadam!(nepochs,ntraj,nsteps,dt,0.0,[0.63,0.03],lrf,lrv,beta1f,beta2f,beta1v,beta2v,eps,system,model,integrator,softmcvb)

## define new value baseline function
#nntmp = Chain(Dense(3,100,x->(2 .* sigmoid.(x) .- 1.0)),
#              Dense(100,100,x->(2 .* sigmoid.(x) .- 1.0)),
#              Dense(100,1))
#theta,re = Flux.destructure(nntmp)
#theta .= 0.0
#nntmp = re(theta)
#vbl = NNValueBaseline(nntmp)
#
## set up optimization loop with hard boundary conditions
#mcvb = MCVBCallback(Afunc,nn,vbl)
#nepochs = 5
#ntraj = 5
#lrf = dt
#lrv = dt
#mcvbtrainsgd!(nepochs,ntraj,nsteps,dt,0.0,[0.63,0.03],lrf,lrv,system,model,integrator,mcvb)
