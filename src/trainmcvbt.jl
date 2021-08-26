using DelimitedFiles
using MPI

function forcecondition(sys::AbstractSystem,t::Float64)
  if sys.t >= t
    return true
  else
    return false
  end
end

function runtrajs!(ntraj::Int64,nsteps::Int64,dt::Float64,t0::Float64,
                   x0::Vector{Float64},system::AbstractThermostattedSystem,
                   integrator::StochasticEuler,mcvb::MCVBCallback)

  # run trajectories for evaluating gradients
  for traj in 1:ntraj
    system.x = copy(x0)
    system.t = t0
    #integrator.model.potentials[end].condition(sys::AbstractSystem) = forcecondition(sys,dt*rand(0:nsteps))
    f(sys::AbstractSystem) = forcecondition(sys,dt*rand(0:nsteps))
    integrator.model.potentials[end].condition = f
    initialize!(mcvb)
    runtraj!(nsteps,dt,integrator,mcvb)
  end

end

"""
Train forces using stochastic gradient descent method with parallelization
"""
# TODO add IO capabilities
function mcvbttrainsgdpar!(nepochs::Int64,totntraj::Int64,nsteps::Int64,dt::Float64,
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
    runtrajs!(ntraj,nsteps,dt,t0,x0,system,integrator,mcvb)

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
        writedlm(Vfile*"_$epoch.txt",mcvb.vbl.theta)
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