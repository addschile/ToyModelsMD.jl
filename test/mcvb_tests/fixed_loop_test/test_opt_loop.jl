include("../../../src/ToyModelsMD.jl")
using .ToyModelsMD
using DelimitedFiles

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
  mut[j] = 1+(j*steps/(Mt-1)) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  mut[j] = (j*steps/(Mt-1)) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
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
x = readdlm("../coeff0x.txt")
y = readdlm("../coeff0y.txt")
cs = zeros(Float64,shape)
for i in 1:21
  cs[:,:,i,1] .= reshape(x[:,i],(21,21))'
  cs[:,:,i,2] .= reshape(y[:,i],(21,21))'
end
cs = reshape(cs, (21*21*21,2))
cs .= 0.0
# define model - yay!
gm = GaussianModel2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,cs)

# mixed model
model = MixedModel(2,[mb,gm])

# thermostat and system
langevin = Langevin(1.0,1.0)
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
  vmut[j] = 1+(j-1)*steps*1.0/(Mt-1) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
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
#v = readdlm("../coeff0V.txt")
vcs = zeros(Float64,shape)
#for i in 1:21
#  vcs[:,:,i] .= reshape(v[:,i],(21,21))'
#end
#vcs = reshape(vcs, (21*21*21))
vbl = GaussianValueBaseline2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,vcs)

# define callback function
mcvb = MCVBCallback(Afunc,gm,vbl)
# set up integrator
integrator = StochasticEuler(system,model)

# set up optimization loop
nepochs = 500
ntraj = 60
lrf = 0.2*dt
lrv = 5.0*dt
dkls = zeros(Float64,nepochs)
avgAs = zeros(Float64,nepochs)
for i in 1:nepochs
  println("Epoch: $i")
  dkl = 0.0
  avgA = 0.0
  gradf = zeros(Float64,2*Mx*My*Mt)
  gradv = zeros(Float64,Mx*My*Mt)
  for j in 1:ntraj
    flush(stdout)
    system.x = [0.63,0.03]
    system.t = 0.0
    initialize!(mcvb)
    runtraj!(steps,dt,integrator,mcvb)
    println("Traj: $j ",Afunc(system,model,1500,dt))
    gradf += mcvb.gradf
    gradv += mcvb.gradv
    dkl += mcvb.dkl
    avgA += Afunc(system,model,1500,dt)
  end
  # average quantities
  gradf /= ntraj
  gradv /= ntraj
  dkl /= ntraj
  avgA /= ntraj
  # compute some stuff
  dkls[i] = dkl
  avgAs[i] = avgA
  println(avgA," ",dkl)
  writedlm("dkls.txt",dkls)
  writedlm("avgAs.txt",avgAs)
  # run some extra trajectories
  xstore = zeros(steps,2)
  cbf(system,args...) = cbfunc(xstore,system,args)
  cb = Callback(1,cbf)
  for j in 1:10
    println("Trajectory $j")
    xstore .= 0.0
    system.x = [0.63,0.03]
    system.t = 0.0
    runtraj!(steps,dt,integrator,cb)
    println(Afunc(system,model,1500,dt)*dt)
    writedlm("traj_$i$j.txt",xstore)
  end
  # update coefficients
  model.potentials[2].cs += (lrf .* reshape(gradf,(Mx*My*Mt,2)))
  mcvb.vbl.cs += (lrv .* gradv)
  writedlm("running_coeffF.txt",model.potentials[2].cs)
  writedlm("running_coeffV.txt",mcvb.vbl.cs)
end

# define callback function
xstore = zeros(steps,2)
cbf(system,args...) = cbfunc(xstore,system,args)
cb = Callback(1,cbf)
for i in 1:100
  println("Trajectory $i")
  flush(stdout)
  xstore .= 0.0
  system.x = [0.63,0.03]
  system.t = 0.0
  runtraj!(steps,dt,integrator,cb)
  println(Afunc(system,model,1500,dt))
  writedlm("traj_final_$i.txt",xstore)
end
