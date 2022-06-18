## TODO adding this for being able to add different optimizers
### TODO add IO capabilities
#"""
#Train forces using stochastic gradient estimation.
#"""
## TODO add parallel version
#function computegrads!(ntraj::Int64,nsteps::Int64,dt::Float64,
#                       t0::Float64,x0::Vector{Float64},
#                       system::AbstractThermostattedSystem,model::MixedModel,
#                       integrator::StochasticEuler,mcvb::MCVBCallback)
#  # zero out the averages
#  initializeavgs!(mcvb)
#  # run trajectories
#  runtrajs!(ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb)
#  # average them ove rthe number of trajectories
#  average!(ntraj,mcvb)
#  # print some useful stuff
#  println(mcvb.dkl," ",mcvb.aval)
#  flush(stdout)
#end
#
#function computeparallelgrads!(totntraj::Int64,nsteps::Int64,dt::Float64,
#                               t0::Float64,x0::Vector{Float64},
#                               system::AbstractThermostattedSystem,model::MixedModel,
#                               integrator::StochasticEuler,mcvb::MCVBCallback)
#
#  # get some MPI stuff
#  comm::MPI.Comm = MPI.COMM_WORLD
#  rank::Int64 = MPI.Comm_rank(comm)
#  nprocs::Int64 = MPI.Comm_size(comm)
#  
#  # calculate number of trajectories for this processor
#  ntraj::Int64 = floor(Int64,totntraj/nprocs)
#  extra::Int64 = totntraj % nprocs 
#  if (nprocs - rank - 1) < extra
#    ntraj += 1
#  end
#
#  if rank == 0
#    gradf_buf::Vector{Float64} = zeros(Float64, size(mcvb.gradf))
#    gradv_buf::Vector{Float64} = zeros(Float64, size(mcvb.gradv))
#  end
#
#  computegrads!(ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb)
#
#  # average the gradients
#  if rank == 0
#
#    # receive information from each sub process and add to average
#    for src in 1:(nprocs-1)
#      MPI.Recv!(gradf_buf,src,100*src+0,comm)
#      mcvb.gradf += gradf_buf
#      MPI.Recv!(gradv_buf,src,100*src+1,comm)
#      mcvb.gradv += gradv_buf
#      mcvb.dkl += MPI.recv(src,100*src+2,comm)[1]
#      mcvb.aval += MPI.recv(src,100*src+3,comm)[1]
#    end
#
#  else
#
#    # send gradients to main processor
#    MPI.Send((ntraj/totntraj).*mcvb.gradf,0,100*rank+0,comm)
#    MPI.Send((ntraj/totntraj).*mcvb.gradv,0,100*rank+1,comm)
#    MPI.send((ntraj/totntraj)*mcvb.dkl,0,100*rank+2,comm)
#    MPI.send((ntraj/totntraj)*mcvb.aval,0,100*rank+3,comm)
#
#  end
# 
#  # broadcast the updated parameters
#  MPI.Barrier(comm)
#  MPI.Bcast!(mcvb.gradf,0,comm)
#  MPI.Bcast!(mcvb.gradv,0,comm)
#  MPI.Barrier(comm)
#
#end
#
##function mcvbtrain!(nepochs::Int64,ntraj::Int64,nsteps::Int64,dt::Float64,
##                    t0::Float64,x0::Vector{Float64},
##                    system::AbstractThermostattedSystem,model::MixedModel,
##                    integrator::StochasticEuler,mcvb::MCVBCallback,
##                    foptimizer::AbstractOptimizer,voptimizer::AbstractOptimizer;
##                    printevery::Int64=1)
##
##  # start optimization
##  for epoch in 1:nepochs
##  
##    # print update
##    println("# Epoch: $epoch")
##    flush(stdout)
##  
##    # compute gradients
##    computegrads!(nepochs,ntraj,nsteps,dt,t0,x0,system,model,integrator,mcvb)
##
##    # update coefficients for forces
##    optimizerstep!(mcvb.gradf,model.potentials[end].theta,foptimizer)
##    optimizerstep!(mcvb.gradv,mcvb.vbl.theta,voptimizer)
##    setparams!(model.potentials[end])
##    setparams!(mcvb.vbl)
##
###    # TODO file system handling and outpout
###    # write updated coefficients
###    writedlm("running_coeffF.txt",model.potentials[end].theta)
###    writedlm("running_coeffV.txt",mcvb.vbl.theta)
###    if epoch % printevery == 0
###      writedlm("coeffF_$epoch.txt",model.potentials[end].theta)
###      writedlm("coeffV_$epoch.txt",mcvb.vbl.theta)
###    end
##  
##  end
##
##end
#