using DelimitedFiles
using MPI

function runtrajs!(ntraj::Int64,nsteps::Int64,dt::Float64,t0::Float64,
                   x0::Vector{Float64},system::AbstractThermostattedSystem,
                   model::MixedModel,integrator::StochasticEuler,
                   mcvb::MCVBCallback)

  # run trajectories for evaluating gradients
  for traj in 1:ntraj
    system.x = copy(x0)
    system.t = t0
    initialize!(mcvb)
    runtraj!(nsteps,dt,integrator,mcvb)
  end

end

"""
"""
function averagegrads!(comm::MPI.Comm,nprocs::Int64,;
                       gradf_buf::Union{Nothing,Vector{Float64}}=nothing,
                       gradv_buf::Union{Nothing,Vector{Float64}}=nothing)

  if rank == 0

    # receive information from each sub process and add to average
    for src in 1:(nprocs-1)
      MPI.Recv!(gradf_buf,src,100*src+0,comm)
      mcvb.gradf += gradf_buf
      MPI.Recv!(gradv_buf,src,100*src+1,comm)
      mcvb.gradv += gradv_buf
      mcvb.dkl += MPI.recv(src,100*src+2,comm)[1]
      mcvb.aval += MPI.recv(src,100*src+3,comm)[1]
    end

  else

    # send gradients to main processor
    MPI.Send(mcvb.gradf,0,100*rank+0,comm)
    MPI.Send(mcvb.gradv,0,100*rank+1,comm)
    MPI.send(mcvb.dkl,0,100*rank+2,comm)
    MPI.send(mcvb.aval,0,100*rank+3,comm)

  end

"""
Train forces using stochastic gradient descent method
"""
function mcvbtrainsgd!(ntraj::Int64,nsteps::Int64,dt::Float64,t0::Float64,
                       x0::Vector{Float64},lrf::Float64,lrv::Float64,
                       system::AbstractThermostattedSystem,model::MixedModel,
                       integrator::StochasticEuler,mcvb::MCVBCallback)

  # zero out the averages
  initializeavgs!(mcvb)

  # run trajectories
  runtrajs!(ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb)

  # average them ove rthe number of trajectories
  average!(ntraj,mcvb)

  # write average quantities to stdout
  println(mcvb.dkl," ",mcvb.aval)
  flush(stdout)
 
  # update coefficients for forces
  updateparams!(lrf,mcvb.gradf,model.potentials[end])

  # update coefficients for valuebaseline
  updateparams!(lrv,mcvb.gradv,mcvb.vbl)

end

"""
Train forces using stochastic gradient descent method with parallelization
"""
# TODO add IO capabilities
function mcvbtrainsgdpar!(ntraj::Int64,nsteps::Int64,dt::Float64,t0::Float64,
                          x0::Vector{Float64},lrf::Float64,lrv::Float64,
                          system::AbstractThermostattedSystem,model::MixedModel,
                          integrator::StochasticEuler,mcvb::MCVBCallback)

  comm::MPI.Comm = MPI.COMM_WORLD
  rank::Int64 = MPI.Comm_rank(comm)
  nprocs::Int64 = MPI.Comm_size(comm)
  
  if rank == 0
    gradf_buf::Vector{Float64} = zeros(Float64, size(mcvb.gradf))
    gradv_buf::Vector{Float64} = zeros(Float64, size(mcvb.gradv))
  else
    gradf_buf::Nothing = nothing
    gradv_buf::Nothing = nothing
  end

  # zero out the averages
  initializeavgs!(mcvb)

  # run trajectories and average gradients
  runtrajs!(ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb)

  # communicate the gradients between processors for averaging
  averagegrads!(comm,nprocs,mcvb,gradf_buf,gradv_bug)

  # average the gradients
  if rank == 0

    # average them over the number of trajectories
    average!(totntraj,mcvb)

    # average them over the number of trajectories
    average!(totntraj,mcvb)

    # TODO make nicer with IO
    # write average quantities to stdout
    println(mcvb.dkl," ",mcvb.aval)
    flush(stdout)

    # update coefficients for forces
    updateparams!(lrf,mcvb.gradf,model.potentials[end])

    # update coefficients for valuebaseline
    updateparams!(lrv,mcvb.gradv,mcvb.vbl)

  else

    # send gradients to main processor
    MPI.Send(mcvb.gradf,0,100*rank+0,comm)
    MPI.Send(mcvb.gradv,0,100*rank+1,comm)
    MPI.send(mcvb.dkl,0,100*rank+2,comm)
    MPI.send(mcvb.aval,0,100*rank+3,comm)

  end
 
  # broadcast the updated parameters
  MPI.Barrier(comm)
  MPI.Bcast!(model.potentials[end].theta,0,comm)
  MPI.Bcast!(mcvb.vbl.theta,0,comm)
  
  # change the parameters for the model on the other processors
  if rank != 0
    setparams!(model.potentials[end])
    setparams!(mcvb.vbl)
  end

  # synchronize
  MPI.Barrier(comm)

end

"""
Train forces using ADAM optimizer
"""
# TODO add IO capabilities
function mcvbtrainadam!(ntraj::Int64,nsteps::Int64,dt::Float64,t0::Float64,
                        x0::Vector{Float64},lrf::Float64,lrv::Float64,
                        beta1f::Float64,beta2f::Float64,beta1v::Float64,
                        beta2v::Float64,eps::Float64,t::Float64,
                        mtf::Vector{Float64},vtf::Vector{Float64},
                        mtv::Vector{Float64},vtv::Vector{Float64},
                        system::AbstractThermostattedSystem,model::MixedModel,
                        integrator::StochasticEuler,mcvb::MCVBCallback)

  # zero out the averages
  initializeavgs!(mcvb)

  # run trajectories
  runtrajs!(ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb)

  # average them ove rthe number of trajectories
  average!(ntraj,mcvb)

  # TODO make nicer with IO
  # write average quantities to stdout
  println(mcvb.dkl," ",mcvb.aval)
  flush(stdout)
 
  # update bias corrections
  mtf .= (beta1f .* mtf) .+ ((1-beta1f) .* mcvb.gradf)
  vtf .= (beta2f .* vtf) .+ ((1-beta2f) .* mcvb.gradf.^2)
  mtv .= (beta1v .* mtv) .+ ((1-beta1v) .* mcvb.gradv)
  vtv .= (beta2v .* vtv) .+ ((1-beta2v) .* mcvb.gradv.^2)

  # add bias correction to gradients
  alphaf = lrf * sqrt(1-beta2f^t) / (1-beta1f^t)
  alphav = lrv * sqrt(1-beta2v^t) / (1-beta1v^t)
  mcvb.gradf .=  mtf ./ (sqrt.(vtf) .+ eps)
  mcvb.gradv .=  mtv ./ (sqrt.(vtv) .+ eps)
  t += 1

  # update coefficients for forces
  updateparams!(lrf,mcvb.gradf,model.potentials[end])

  # update coefficients for valuebaseline
  updateparams!(lrv,mcvb.gradv,mcvb.vbl)

end

"""
Train forces using stochastic gradient descent method with parallelization
"""
# TODO add IO capabilities
function mcvbtrainadampar!(ntraj::Int64,nsteps::Int64,dt::Float64,t0::Float64,
                           x0::Vector{Float64},lrf::Float64,lrv::Float64,
                           beta1f::Float64,beta2f::Float64,beta1v::Float64,
                           beta2v::Float64,eps::Float64,t::Float64,
                           system::AbstractThermostattedSystem,model::MixedModel,
                           integrator::StochasticEuler,mcvb::MCVBCallback;
                           mtf::Union{Nothing,Vector{Float64}}=nothing,
                           vtf::Union{Nothing,Vector{Float64}}=nothing,
                           mtv::Union{Nothing,Vector{Float64}}=nothing,
                           vtv::Union{Nothing,Vector{Float64}}=nothing)

  comm::MPI.Comm = MPI.COMM_WORLD
  rank::Int64 = MPI.Comm_rank(comm)
  nprocs::Int64 = MPI.Comm_size(comm)
  
  # calculate number of trajectories for this processor
  ntraj::Int64 = floor(Int64,totntraj/nprocs)
  extra::Int64 = totntraj % nprocs 
  if (nprocs - rank - 1) < extra
    ntraj += 1
  end

  if rank == 0
    gradf_buf::Vector{Float64} = zeros(Float64, size(mcvb.gradf))
    gradv_buf::Vector{Float64} = zeros(Float64, size(mcvb.gradv))
  end

  # start optimization
  for epoch in 1:nepochs
  
    # print update
    if rank == 0
      println("# Epoch: $epoch")
      flush(stdout)
    end
  
    # zero out the averages
    initializeavgs!(mcvb)

    # run trajectories and average gradients
    runtrajs!(ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb)

    # average the gradients
    if rank == 0

      # receive information from each sub process and add to average
      for src in 1:(nprocs-1)
        MPI.Recv!(gradf_buf,src,100*src+0,comm)
        mcvb.gradf += gradf_buf
        MPI.Recv!(gradv_buf,src,100*src+1,comm)
        mcvb.gradv += gradv_buf
        mcvb.dkl += MPI.recv(src,100*src+2,comm)[1]
        mcvb.aval += MPI.recv(src,100*src+3,comm)[1]
      end

      # average them over the number of trajectories
      average!(totntraj,mcvb)

      # TODO make nicer with IO
      # write average quantities to stdout
      println(mcvb.dkl," ",mcvb.aval)
      flush(stdout)

      # update bias corrections
      mtf .= (beta1f .* mtf) .+ ((1-beta1f) .* mcvb.gradf)
      vtf .= (beta2f .* vtf) .+ ((1-beta2f) .* mcvb.gradf.^2)
      mtv .= (beta1v .* mtv) .+ ((1-beta1v) .* mcvb.gradv)
      vtv .= (beta2v .* vtv) .+ ((1-beta2v) .* mcvb.gradv.^2)

      # add bias correction to gradients
      mhatf .= mtf ./ (1-beta1f^t)
      vhatf .= vtf ./ (1-beta2f^t)
      mhatv .= mtv ./ (1-beta1v^t)
      vhatv .= vtv ./ (1-beta2v^t)
      mcvb.gradf .=  - mhatf ./ (sqrt.(vhatf) .+ eps)
      mcvb.gradv .=  - mhatv ./ (sqrt.(vhatv) .+ eps)
      t += 1

      # update coefficients for forces
      updateparams!(lrf,mcvb.gradf,model.potentials[end])

      # update coefficients for valuebaseline
      updateparams!(lrv,mcvb.gradv,mcvb.vbl)

    else

      # send gradients to main processor
      MPI.Send(mcvb.gradf,0,100*rank+0,comm)
      MPI.Send(mcvb.gradv,0,100*rank+1,comm)
      MPI.send(mcvb.dkl,0,100*rank+2,comm)
      MPI.send(mcvb.aval,0,100*rank+3,comm)

    end
 
    # broadcast the updated parameters
    MPI.Barrier(comm)
    MPI.Bcast!(model.potentials[end].theta,0,comm)
    MPI.Bcast!(mcvb.vbl.theta,0,comm)
    
    # change the parameters for the model on the other processors
    if rank != 0
      setparams!(model.potentials[end])
      setparams!(mcvb.vbl)
    end

    # synchronize
    MPI.Barrier(comm)

  end

end

function mcvbtrain!(nepochs::Int64,totntraj::Int64,nsteps::Int64,dt::Float64,
                    t0::Float64,x0::Vector{Float64},hyperparams::Vector{Float64},
                    system::AbstractThermostattedSystem,model::MixedModel,
                    integrator::StochasticEuler,mcvb::MCVBCallback;
                    optimizer::String="sgd",varlr::Bool=False,
                    lrscale::Union{Float64,Function}=1.0,lrscaleevery::Int64=1000,
                    restart=true,printevery::Int64=1,Ffile::String="coeffF",
                    Vfile::String="coeffF")

  # do some MPI stuff
  if MPI.Initialized() == false
    MPI.Init()
  end
  comm::MPI.Comm = MPI.COMM_WORLD
  rank::Int64 = MPI.Comm_rank(comm)
  nprocs::Int64 = MPI.Comm_size(comm)
  
  # calculate number of trajectories for this processor
  ntraj::Int64 = floor(Int64,totntraj/nprocs)
  extra::Int64 = totntraj % nprocs 
  if (nprocs - rank - 1) < extra
    ntraj += 1
  end

  # get learning rates
  if optimizer == "sgd"
    @assert length(hyperparams) == 2 "Number of hyperparameters should be 2, but given $(length(hyperparams))"
    lrf::Float64 = hyperparams[1]
    lrv::Float64 = hyperparams[2]
  else if optimizer == "adam"
    @assert length(hyperparams) == 6 "Number of hyperparameters should be 6, but given $(length(hyperparams))"
    lrf::Float64    = hyperparams[1]
    beta1f::Float64 = hyperparams[2]
    beta2f::Float64 = hyperparams[3]
    lrv::Float64    = hyperparams[4]
    beta1v::Float64 = hyperparams[5]
    beta2v::Float64 = hyperparams[6]
    # initialize extra vectors for adam algorithm
    mtf::Vector{Float64} = zeros(Float64, size(mcvb.gradf))
    vtf::Vector{Float64} = zeros(Float64, size(mcvb.gradf))
    mtv::Vector{Float64} = zeros(Float64, size(mcvb.gradv))
    vtv::Vector{Float64} = zeros(Float64, size(mcvb.gradv))
    t::Int64 = 1
  end

  # start optimization
  for epoch in 1:nepochs
  
    # print update
    if rank == 0
      println("# Epoch: $epoch")
      flush(stdout)
    end
  
    if nprocs == 1
      if optimizer == "sgd"
        mcvbtrainsgd!(ntraj,nsteps,dt,t0,x0,lrf,lrv,system,model,integrator,mcvb)
      else if optimizer == "adam"
        mcvbtrainadam!(ntraj,nsteps,dt,t0,x0,lrf,lrv,beta1f,beta2f,beta1v,beta2v,
                       eps,t,mtf,vtf,mtv,vtv,system,model,integrator,mcvb)
      end
    end
    #else
    #  if optimizer == "sgd"
    #    mcvbtrainsgdpar!(ntraj,nsteps,dt,t0,x0,lrf,lrv,system,model,integrator,mcvb)
    #  else if optimizer == "adam"
    #     if rank == 0
    #     mcvbtrainadampar!(ntraj,nsteps,dt,t0,x0,lrf,lrv,beta1f,beta2f,beta1v,
    #                       beta2v,eps,t,system,model,integrator,mcvb;
    #                       mtf=mtf,vtf=vtf,mtv=mtv,vtv=vtv)
    #     else
    #     mcvbtrainadampar!(ntraj,nsteps,dt,t0,x0,lrf,lrv,beta1f,beta2f,beta1v,
    #                       beta2v,eps,t,system,model,integrator,mcvb)
    #     end
    #  end
    #end

    # TODO file system handling and outpout
    if restart
      writedlm("running_coeffF.txt",model.potentials[end].theta)
      writedlm("running_coeffV.txt",mcvb.vbl.theta)
    end

    if epoch % printevery == 0
      # write updated coefficients
      writedlm(Ffile*"_$epoch.txt",model.potentials[end].theta)
      writedlm(Vfile*"_$epoch.txt",model.potentials[end].theta)
    end

    # TODO implement learning rate updates
    # update learning rates if requested
    if varlr
      if epoch % lrscaleevery == 0
        lrf *= lrscale
        lrv *= lrscale
      end
    end

  end

end