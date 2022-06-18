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
steps = 1500
tf = steps*dt

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
nepochs = 50
ntraj = 5#20
lrf = dt/50
lrv = dt/50
dkls = zeros(Float64,nepochs)
avgAs = zeros(Float64,nepochs)
savgAs = zeros(Float64,nepochs)
for i in 1:nepochs
  println("==> Epoch: $i <==")
  flush(stdout)
  dkl = 0.0
  avgA = 0.0
  savgA = 0.0
  gradf = zeros(Float64,size(softmcvb.gradf))
  gradv = zeros(Float64,size(softmcvb.gradv))
  for j in 1:ntraj
    system.x = [0.63,0.03]
    system.t = 0.0
    initializeall!(softmcvb)
    runtraj!(steps,dt,integrator,softmcvb)
    println("Traj: $j, ",system.x," ",softAfunc(system,model,1500,dt)," ",Afunc(system,model,1500,dt))
    flush(stdout)
    gradf += softmcvb.gradf
    gradv += softmcvb.gradv
    dkl += softmcvb.dkl
    avgA += Afunc(system,model,1500,dt)
    savgA += softAfunc(system,model,1500,dt)
  end
  # average quantities
  gradf /= ntraj
  gradv /= ntraj
  dkl /= ntraj
  avgA /= ntraj
  savgA /= ntraj
  # compute some stuff
  dkls[i] = dkl
  avgAs[i] = avgA
  savgAs[i] = savgA
  println(avgA," ",savgA," ",dkl)
  writedlm("init_dkls.txt",dkls)
  writedlm("init_avgAs.txt",avgAs)
  writedlm("init_savgAs.txt",savgAs)
  ## run some extra trajectories
  #xstore = zeros(steps,2)
  #cbf(system,args...) = cbfunc(xstore,system,args)
  #cb = Callback(1,cbf)
  #for j in 1:10
  #  println("Trajectory $j")
  #  xstore .= 0.0
  #  system.x = [0.63,0.03]
  #  system.t = 0.0
  #  runtraj!(steps,dt,integrator,cb)
  #  println(Afunc(system,model,1500,dt))
  #  writedlm("init_traj_$i$j.txt",xstore)
  #end
  # update coefficients for forces
  theta,re = Flux.destructure(model.potentials[2].nn)
  theta .+= (lrf .* gradf)
  writedlm("init_running_coeffF.txt",theta)
  model.potentials[2].nn = re(theta) 
  model.potentials[2].pars = params(model.potentials[2].nn)
  # update coefficients for value baseline
  theta,re = Flux.destructure(softmcvb.vbl.nn)
  theta .+= (lrv .* gradv)
  writedlm("init_running_coeffV.txt",theta)
  softmcvb.vbl.nn = re(theta)
  softmcvb.vbl.pars = params(softmcvb.vbl.nn)
end

## set up final optimization loop
#nntmp = Chain(Dense(3,100,x->(2 .* sigmoid.(x) .- 1.0)),
#              Dense(100,100,x->(2 .* sigmoid.(x) .- 1.0)),
#              Dense(100,1))
#theta,re = Flux.destructure(nntmp)
#theta .= 0.0
#nntmp = re(theta)
#vbl = NNValueBaseline(nntmp)
#mcvb = MCVBCallback(Afunc,nn,vbl)
#nepochs = 500
#ntraj = 60
#lrf = dt
#lrv = dt
#dkls = zeros(Float64,nepochs)
#avgAs = zeros(Float64,nepochs)
#for i in 1:nepochs
#  println("==> Epoch: $i <==")
#  dkl = 0.0
#  avgA = 0.0
#  gradf = zeros(Float64,size(mcvb.gradf))
#  gradv = zeros(Float64,size(mcvb.gradv))
#  for j in 1:ntraj
#    system.x = [0.63,0.03]
#    system.t = 0.0
#    initialize!(mcvb)
#    runtraj!(steps,dt,integrator,mcvb)
#    println("Traj: $j, ",system.x," ",Afunc(system,model,1500,dt))
#    gradf += mcvb.gradf
#    gradv += mcvb.gradv
#    dkl += mcvb.dkl
#    avgA += Afunc(system,model,1500,dt)
#  end
#  # average quantities
#  gradf /= ntraj
#  gradv /= ntraj
#  dkl /= ntraj
#  avgA /= ntraj
#  # compute some stuff
#  dkls[i] = dkl
#  avgAs[i] = avgA
#  println(avgA," ",dkl)
#  writedlm("dkls.txt",dkls)
#  writedlm("avgAs.txt",avgAs)
#  # run some extra trajectories
#  xstore = zeros(steps,2)
#  cbf(system,args...) = cbfunc(xstore,system,args)
#  cb = Callback(1,cbf)
#  for j in 1:10
#    println("Trajectory $j")
#    xstore .= 0.0
#    system.x = [0.63,0.03]
#    system.t = 0.0
#    runtraj!(steps,dt,integrator,cb)
#    println(Afunc(system,model,1500,dt))
#    writedlm("opt_traj_$i$j.txt",xstore)
#  end
#  # update coefficients for forces
#  theta,re = Flux.destructure(model.potentials[2].nn)
#  theta .+= (lrf .* gradf)
#  writedlm("running_coeffF.txt",theta)
#  model.potentials[2].nn = re(theta) 
#  model.potentials[2].pars = params(model.potentials[2].nn)
#  # update coefficients for value baseline
#  theta,re = Flux.destructure(mcvb.vbl.nn)
#  theta .+= (lrv .* gradv)
#  writedlm("running_coeffV.txt",theta)
#  softmcvb.vbl.nn = re(theta)
#  softmcvb.vbl.pars = params(softmcvb.vbl.nn)
#end
#
## define callback function
#xstore = zeros(steps,2)
#cbf(system,args...) = cbfunc(xstore,system,args)
#cb = Callback(1,cbf)
#for i in 1:10
#  println("Trajectory $i")
#  xstore .= 0.0
#  system.x = [0.63,0.03]
#  system.t = 0.0
#  runtraj!(steps,dt,integrator,cb)
#  println(Afunc(system,model,1500,dt))
#  writedlm("final_traj_$i.txt",xstore)
#end
