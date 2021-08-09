include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using DelimitedFiles
using Plots

function cbfunc(xstore::Array,system::AbstractSystem,args::Tuple{MixedModel, Tuple{Int64, Float64}})
  xstore[args[2][1],:] .= copy(system.x)
end

### define system ###
dt = 0.0001
steps = 1500
tf = steps*dt

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
  mut[j] = 1+(j-1)*steps*1.0/(Mt-1) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  mut[j] *= dt
end
sigx = zeros(Float64,Mx)
sigy = zeros(Float64,My)
sigt = zeros(Float64,Mt)
sigx[:] .= ((1.5-(-1.5))/(Mx-1)/2.0)^2 #variance
sigy[:] .= ((2-(-0.5))/(My-1)/2.0)^2 #variance
sigt[:] .= ((steps*1.0/(Mt-1))/2.0)^2 #variance
sigt[:] .*= dt^2
# read in optimal coefficients
x = readdlm("coeff0xn.txt")
y = readdlm("coeff0yn.txt")
cs = zeros(Float64,shape)
for i in 1:21
  cs[:,:,i,1] .= reshape(x[:,i],(21,21))'
  cs[:,:,i,2] .= reshape(y[:,i],(21,21))'
end
cs = reshape(cs,(21*21*21,2))
# define model - yay!
gm = GaussianModel2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,cs)

# mixed model
model = MixedModel(2,[mb,gm])

# thermostat and system
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

# define callback function
xstore = zeros(steps,2)
cbf(system,args...) = cbfunc(xstore,system,args)
cb = Callback(1,cbf)
# set up integrator and run!
integrator = StochasticEuler(system,model)
runtraj!(steps,dt,integrator,cb)
writedlm("traj.txt",xstore)

@time for i in 1:10
  println("Trajectory $i")
  xstore .= 0.0
  integrator.system.x = [0.63,0.03]
  integrator.system.t = 0.0
  runtraj!(steps,dt,integrator,cb)
  writedlm("traj_$i.txt",xstore)
end
