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
Mx = 21
My = 21
Mt = 21
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
  vmut[j] = (j*steps*1.0/(Mt-1)) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  vmut[j] *= dt
end
vsigx = zeros(Float64,Mx)
vsigy = zeros(Float64,My)
vsigt = zeros(Float64,Mt)
vsigx[:] .= ((1.5-(-1.5))/(Mx-1)/2.0)^2 #variance
vsigy[:] .= ((2-(-0.5))/(My-1)/2.0)^2 #variance
vsigt[:] .= ((steps*1.0/(Mt-1))/2.0)^2 #variance
vsigt[:] .*= dt^2
# read in optimal coefficients
v = readdlm("coeff0V.txt")
vcs = zeros(Float64,shape)
for i in 1:21
  vcs[:,:,i] .= reshape(v[:,i],(21,21))'
end
vcs = reshape(vcs, (21*21*21))
vbl = GaussianValueBaseline2D(Mx,My,Mt,vmux,vmuy,vmut,vsigx,vsigy,vsigt,vcs)

# define callback function
mcvb = MCVBCallback(Afunc,nn,vbl)
# set up integrator and run!
integrator = StochasticEuler(system,model)
runtraj!(steps,dt,integrator,mcvb)
println(mcvb.dkl)
writedlm("gradf.txt",mcvb.gradf)
writedlm("gradv.txt",mcvb.gradv*dt)
