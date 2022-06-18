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
  lam::Float64 = 2.0e3/dt
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

# gaussian force model
Mx = 21
My = 21
Mt = 21
shape = (Mx,My,Mt,2)
# Define gaussian basis centers and variance
mux = zeros(Float64,Mx)
muy = zeros(Float64,My)
mut = zeros(Float64,Mt)
for j=1:Mx
  mux[j] = -1.5+(j-1)*(1.5-(-1.5))/(Mx-1)
end
for j=1:My
  muy[j] = -0.5+(j-1)*(2-(-0.5))/(My-1)
end
for j in 1:Mt
  mut[j] = 1+(j-1)*nsteps*1.0/(Mt-1) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  mut[j] *= dt
end
sigx = zeros(Float64,Mx)
sigy = zeros(Float64,My)
sigt = zeros(Float64,Mt)
sigx[:] .= ((1.5-(-1.5))/(Mx-1)/2.0)^2 #variance
sigy[:] .= ((2-(-0.5))/(My-1)/2.0)^2 #variance
sigt[:] .= ((nsteps*1.0/(Mt-1))/2.0)^2 #variance
sigt[:] .*= dt^2
# initialize coefficients
theta = zeros(Float64,(21*21*21,2))
# define model - yay!
gm = GaussianModel2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,theta)

# mixed model
model = MixedModel(2,[mb,gm])

# thermostat and system
langevin = Langevin(1.0,1.0,convert(UInt64,42))
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

# define value baseline function
shape = (Mx,My,Mt)
vmux = zeros(Float64,Mx)
vmuy = zeros(Float64,My)
vmut = zeros(Float64,Mt)
for j=1:Mx
  vmux[j] = -1.5+(j-1)*(1.5-(-1.5))/(Mx-1)
end
for j=1:My
  vmuy[j] = -0.5+(j-1)*(2-(-0.5))/(My-1)
end
for j in 1:Mt
  vmut[j] = 1+(j-1)*nsteps*1.0/(Mt-1) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  vmut[j] *= dt
end
vsigx = zeros(Float64,Mx)
vsigy = zeros(Float64,My)
vsigt = zeros(Float64,Mt)
vsigx[:] .= ((1.5-(-1.5))/(Mx-1)/2.0)^2 #variance
vsigy[:] .= ((2-(-0.5))/(My-1)/2.0)^2 #variance
vsigt[:] .= ((nsteps*1.0/(Mt-1))/2.0)^2 #variance
vsigt[:] .*= dt^2
vcs = zeros(Float64, (21*21*21))
vbl = GaussianValueBaseline2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,vcs)

# set up integrator
integrator = StochasticEuler(system,model)

# set up optimization loop with soft boundary conditions
softmcvb = MCVBCallback(softAfunc,gm,vbl)
nepochs = 5#0
ntraj = 10
lrf = dt
lrv = dt
mcvbtrainsgd!(nepochs,ntraj,nsteps,dt,0.0,[0.63,0.03],lrf,lrv,system,model,integrator,softmcvb)

## define new value baseline function
#vcs = zeros(Float64, (21*21*21))
#vbl = GaussianValueBaseline2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,vcs)
#
## set up optimization loop with hard boundary conditions
#mcvb = MCVBCallback(Afunc,gm,vbl)
#nepochs = 5
#ntraj = 5
#lrf = dt
#lrv = dt
#mcvbtrainsgd!(nepochs,ntraj,nsteps,dt,0.0,[0.63,0.03],lrf,lrv,system,model,integrator,mcvb)
