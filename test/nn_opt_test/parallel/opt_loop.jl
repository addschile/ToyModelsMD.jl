using MPI
include("../../../src/ToyModelsMD.jl")
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

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

### define system ###
dt = 0.0001
steps = 1500
tf = steps*dt
ntrajperproc = 5
ntraj = ntrajperproc*nprocs

### define system ###
# mueller-brown potential
mb = MullerBrown()
# neural network model
nntmp = Chain(Dense(3,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,2))
thetaf,ref = Flux.destructure(nntmp)
if rank == 0
  thetaf .= 0.0
end
MPI.Barrier(comm)
MPI.Bcast!(thetaf,0,comm)
nntmp = ref(thetaf)
nn = NNModel(2,nntmp)
# mixed model
model = MixedModel(2,[mb,nn])

# thermostat and system
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

# define value baseline function
nntmpv = Chain(Dense(3,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,100,x->(2 .* sigmoid.(x) .- 1.0)),
              Dense(100,1))
thetav,rev = Flux.destructure(nntmpv)
if rank == 0
  thetav .= 0.0
end
MPI.Barrier(comm)
MPI.Bcast!(thetav,0,comm)
nntmpv = rev(thetav)
vbl = NNValueBaseline(nntmpv)

# set up integrator
integrator = StochasticEuler(system,model)

# set up mcvb callback with softened boundary conditions
softmcvb = MCVBCallback(softAfunc,nn,vbl)

# define some quantities for gradient update and averaging
nepochs = 10
lrf = dt/50
lrv = dt/50
if rank == 0
  dkls = zeros(Float64,nepochs)
  avgAs = zeros(Float64,nepochs)
  savgAs = zeros(Float64,nepochs)
  dkl_buf = 0.0
  avgA_buf = 0.0
  savgA_buf = 0.0
  gradf_buf = zeros(Float64,size(softmcvb.gradf))
  gradv_buf = zeros(Float64,size(softmcvb.gradv))
end

# MPI thing
tag = 0

function runtrajs!(ntraj::Int64,nsteps::Int64,dt::Float64,t0::Float64,
                   x0::Vector{Float64},system::AbstractThermostattedSystem,
                   model::AbstractPotential,integrator::StochasticEuler,
                   mcvb::MCVBCallback,avgs::Vector{Any})

  # zero out the quantities to be averaged
  avgs[1]  = 0.0
  avgs[2]  = 0.0
  avgs[3] .= 0.0
  avgs[4] .= 0.0

  # run trajectories for evaluating gradients
  for j in 1:ntraj
    system.x = copy(x0)
    system.t = t0
    initialize!(mcvb)
    runtraj!(nsteps,dt,integrator,mcvb)
    avgs[1] += mcvb.dkl
    avgs[2] += mcvb.Afunc(system,model,nsteps,dt)
    avgs[3] += mcvb.gradf
    avgs[4] += mcvb.gradv
  end

  # average quantities over number of trajectories
  avgs[1] /= ntraj
  avgs[2] /= ntraj
  avgs[3] /= ntraj
  avgs[4] /= ntraj

end

function graddescent(nepochs::Int64,ntraj::Int64,nsteps::Int64,dt::Float64,
                     t0::Float64,x0::Vector{Float64},lrf::Float64,lrv::Float64,
                     system::AbstractThermostattedSystem,model::AbstractPotential,
                     integrator::StochasticEuler,mcvb::MCVBCallback)

  # start optimization
  for i in 1:nepochs
  
    # print update
    println("==> Epoch: $i <==")
    flush(stdout)
  
    # set up average quantities
    avgs = [0.0,0.0,zeros(Float64,size(mcvb.gradf)),zeros(Float64,size(mcvb.gradv))]

    # run trajectories and average gradients
    runtrajs!(ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb,avgs)

    writedlm("init_dkls.txt",dkls)
    writedlm("init_avgAs.txt",avgAs)
 
    # update coefficients for forces
    global thetaf,ref = Flux.destructure(model.potentials[2].nn)
    thetaf .+= (lrf .* gradf)
    model.potentials[2].nn = ref(thetaf)
    model.potentials[2].pars = params(model.potentials[2].nn)
    if rank == 0
      writedlm("init_running_coeffF.txt",thetaf)
      if i % 5 == 0
        writedlm("init_running_coeffF_$i.txt",thetaf)
      end
    end
  
    # update coefficients for value baseline
    global thetav,rev = Flux.destructure(softmcvb.vbl.nn)
    thetav .+= (lrv .* gradv)
    softmcvb.vbl.nn = rev(thetav)
    softmcvb.vbl.pars = params(softmcvb.vbl.nn)
    if rank == 0
      writedlm("init_running_coeffV.txt",thetav)
      if i % 5 == 0
        writedlm("init_running_coeffV_$i.txt",thetav)
      end
    end
  
  end

end

function optloop(nepochs::Int64,t0::Float64,x0::Vector{Float64},)

  # start optimization
  for i in 1:nepochs
  
    # print update
    if rank == 0
      println("==> Epoch: $i <==")
      flush(stdout)
    end
  
    # zero out the average quantities
    global dkl = 0.0
    global avgA = 0.0
    global savgA = 0.0
    global gradf .= 0.0
    global gradv .= 0.0
  
    # run trajectories for evaluating gradients
    for j in 1:ntrajperproc
      system.x = [0.63,0.03]
      system.t = 0.0
      initialize!(softmcvb)
      runtraj!(steps,dt,integrator,softmcvb)
      println("Traj: $(rank*ntrajperproc + j), ",system.x," ",softAfunc(system,model,1500,dt)," ",Afunc(system,model,1500,dt))
      gradf += softmcvb.gradf
      gradv += softmcvb.gradv
      dkl += softmcvb.dkl
      avgA += Afunc(system,model,1500,dt)
      savgA += softAfunc(system,model,1500,dt)
    end
  
    # average the gradients
    if rank == 0
      # receive information from each sub process and add to average
      for src in 1:(nprocs-1)
        MPI.Recv!(gradf_buf,src,100*src+0,comm)
        gradf += gradf_buf
        MPI.Recv!(gradv_buf,src,100*src+1,comm)
        gradv += gradv_buf
        dkl += MPI.recv(src,100*src+2,comm)[1]
        avgA += MPI.recv(src,100*src+3,comm)[1]
        savgA += MPI.recv(src,100*src+4,comm)[1]
      end
  
      # average quantities over number of trajectories
      gradf /= ntraj
      gradv /= ntraj
      dkl /= ntraj
      avgA /= ntraj
      savgA /= ntraj
  
      # output the updated average quantities
      dkls[i] = dkl
      avgAs[i] = avgA
      savgAs[i] = savgA
      println(avgA," ",savgA," ",dkl)
      writedlm("init_dkls.txt",dkls)
      writedlm("init_avgAs.txt",avgAs)
      writedlm("init_savgAs.txt",savgAs)
  
    else
  
      # send gradients to main processor
      MPI.Send(gradf,0,100*rank+0,comm)
      MPI.Send(gradv,0,100*rank+1,comm)
      MPI.send(dkl,0,100*rank+2,comm)
      MPI.send(avgA,0,100*rank+3,comm)
      MPI.send(savgA,0,100*rank+4,comm)
  
    end
  
    # broadcast the averaged gradients
    MPI.Barrier(comm)
    MPI.Bcast!(gradf,0,comm)
    MPI.Bcast!(gradv,0,comm)
    MPI.Barrier(comm)
  
    # update coefficients for forces
    global thetaf,ref = Flux.destructure(model.potentials[2].nn)
    thetaf .+= (lrf .* gradf)
    model.potentials[2].nn = ref(thetaf)
    model.potentials[2].pars = params(model.potentials[2].nn)
    if rank == 0
      writedlm("init_running_coeffF.txt",thetaf)
      if i % 5 == 0
        writedlm("init_running_coeffF_$i.txt",thetaf)
      end
    end
  
    # update coefficients for value baseline
    global thetav,rev = Flux.destructure(softmcvb.vbl.nn)
    thetav .+= (lrv .* gradv)
    softmcvb.vbl.nn = rev(thetav)
    softmcvb.vbl.pars = params(softmcvb.vbl.nn)
    if rank == 0
      writedlm("init_running_coeffV.txt",thetav)
      if i % 5 == 0
        writedlm("init_running_coeffV_$i.txt",thetav)
      end
    end
  
  end

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
