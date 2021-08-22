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
    #println("Traj: $traj, ",system.x," ",mcvb.Afunc(system,model,1500,dt))
  end

end

"""
Train forces using stochastic gradient descent method
"""
# TODO add IO capabilities
function mcvbtrainsgd!(nepochs::Int64,ntraj::Int64,nsteps::Int64,dt::Float64,
                       t0::Float64,x0::Vector{Float64},lrf::Float64,lrv::Float64,
                       system::AbstractThermostattedSystem,model::MixedModel,
                       integrator::StochasticEuler,mcvb::MCVBCallback;
#                       varlr::Bool=false,lrscale::Float64=1.0,lrscaleevery::Int64=1000,
                       restart=true,printevery::Int64=1,Ffile::String="coeffF",
                       Vfile::String="coeffV")

  # start optimization
  for epoch in 1:nepochs
  
    # print update
    println("# Epoch: $epoch")
    flush(stdout)
  
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
 
    # update coefficients for forces
    updateparams!(lrf,mcvb.gradf,model.potentials[end])

    # update coefficients for valuebaseline
    updateparams!(lrv,mcvb.gradv,mcvb.vbl)

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

#    # update learning rate
#    if varlr
#      if epoch % lrscaleevery
#        lrf *= lrscale
#        lrv *= lrscale
#      end
#    end
  
  end

end

"""
Train forces using stochastic gradient descent method with parallelization
"""
# TODO add IO capabilities
function mcvbtrainsgdpar!(nepochs::Int64,totntraj::Int64,nsteps::Int64,dt::Float64,
                          t0::Float64,x0::Vector{Float64},lrf::Float64,lrv::Float64,
                          system::AbstractThermostattedSystem,model::MixedModel,
                          integrator::StochasticEuler,mcvb::MCVBCallback;
                          restart=true,printevery::Int64=1,Ffile::String="coeffF",
                          Vfile::String="coeffV")

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

      # update coefficients for forces
      updateparams!(lrf,mcvb.gradf,model.potentials[end])

      # update coefficients for valuebaseline
      updateparams!(lrv,mcvb.gradv,mcvb.vbl)

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
    MPI.Barrier(comm)
    
    # change the parameters for the model on the other processors
    if rank != 0
      setparams!(model.potentials[end])
      setparams!(mcvb.vbl)
    end

    # synchronize
    MPI.Barrier(comm)

#    # update learning rate
#    if varlr
#      if epoch % lrscaleevery
#        lrf *= lrscale
#        lrv *= lrscale
#      end
#    end
  
  end

end

"""
Train forces using ADAM optimizer
"""
# TODO add IO capabilities
function mcvbtrainadam!(nepochs::Int64,ntraj::Int64,nsteps::Int64,dt::Float64,
                        t0::Float64,x0::Vector{Float64},lrf::Float64,lrv::Float64,
                        beta1f::Float64,beta2f::Float64,beta1v::Float64,beta2v::Float64,eps::Float64,
                        system::AbstractThermostattedSystem,model::MixedModel,
                        integrator::StochasticEuler,mcvb::MCVBCallback;
                        restart=true,printevery::Int64=1,Ffile::String="coeffF",
                        Vfile::String="coeffV")

  # initialize extra vectors for adam algorithm
  mtf::Vector{Float64}   = zeros(Float64, size(mcvb.gradf))
  vtf::Vector{Float64}   = zeros(Float64, size(mcvb.gradf))
  mtv::Vector{Float64}   = zeros(Float64, size(mcvb.gradv))
  vtv::Vector{Float64}   = zeros(Float64, size(mcvb.gradv))
  t::Int64 = 1

  # start optimization
  for epoch in 1:nepochs
  
    # print update
    println("# Epoch: $epoch")
    flush(stdout)
  
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
  
#    # update learning rate
#    if varlr
#      if epoch % lrscaleevery
#        lrf *= lrscale
#        lrv *= lrscale
#      end
#    end

  end

end

"""
Train forces using stochastic gradient descent method with parallelization
"""
# TODO add IO capabilities
function mcvbtrainadampar!(nepochs::Int64,totntraj::Int64,nsteps::Int64,dt::Float64,
                          t0::Float64,x0::Vector{Float64},lrf::Float64,lrv::Float64,
                          beta1f::Float64,beta2f::Float64,beta1v::Float64,beta2v::Float64,eps::Float64,
                          system::AbstractThermostattedSystem,model::MixedModel,
                          integrator::StochasticEuler,mcvb::MCVBCallback;
                          restart=true,printevery::Int64=1,Ffile::String="coeffF",
                          Vfile::String="coeffV")

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
      alphaf = lrf * sqrt(1-beta2f^t) / (1-beta1f^t)
      alphav = lrv * sqrt(1-beta2v^t) / (1-beta1v^t)
      mcvb.gradf .=  mtf ./ (sqrt.(vtf) .+ eps)
      mcvb.gradv .=  mtv ./ (sqrt.(vtv) .+ eps)
      t += 1

      # update coefficients for forces
      updateparams!(lrf,mcvb.gradf,model.potentials[end])

      # update coefficients for valuebaseline
      updateparams!(lrv,mcvb.gradv,mcvb.vbl)

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
    MPI.Barrier(comm)
    
    # change the parameters for the model on the other processors
    if rank != 0
      setparams!(model.potentials[end])
      setparams!(mcvb.vbl)
    end

    # synchronize
    MPI.Barrier(comm)

  end

end
