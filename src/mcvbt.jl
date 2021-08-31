mutable struct MCVBTCallback <: AbstractCallback
    every::Int64
    dkl::Float64
    aval::Float64
    # biasing observables
    Afunc::Function
    #Bfunc::Function
    # force Malliavin weights
    mvy::Array{Float64}
    mvydot::Array{Float64}
    gradf::Array{Float64}
    # value baseline Malliavin weights
    mvz::Array{Float64}
    mvzdot::Array{Float64}
    gradv::Array{Float64}
    # value baseline function
    vbl::AbstractValueBaseline
    # extra memory for computing some stuff
    em::Array{Float64}
    # IO stuff for writing things to files
    #avgsio::Union{IO,nothing}
    function MCVBTCallback(Afunc::Function,model::AbstractPotential,vbl::AbstractValueBaseline)
      mvy::Array{Float64} = zeros(Float64,gradshape(model))
      mvydot::Array{Float64} = zeros(Float64,jacshape(model))
      gradf::Array{Float64} = zeros(Float64,gradshape(model))
      mvz::Array{Float64} = zeros(Float64,gradshape(vbl))
      mvzdot::Array{Float64} = zeros(Float64,gradshape(vbl))
      gradv::Array{Float64} = zeros(Float64,gradshape(vbl))
      em = zeros(Float64,getdimensionality(model))
      new(1,0.0,0.0,Afunc,mvy,mvydot,gradf,mvz,mvzdot,gradv,vbl,em)
    end
  end
  
  """
  Function that zeros all the quantities for restarting trajectories
  """
  function initialize!(cb::MCVBTCallback)
    cb.mvy .= 0.0
    cb.mvydot .= 0.0
    cb.mvz .= 0.0
    cb.mvzdot .= 0.0
  end
  
  function initializeavgs!(cb::MCVBTCallback)
    cb.dkl = 0.0
    cb.aval = 0.0
    cb.gradf .= 0.0
    cb.gradv .= 0.0
  end
  
  function initializeall!(cb::MCVBTCallback)
    cb.dkl = 0.0
    cb.aval = 0.0
    cb.mvy .= 0.0
    cb.mvydot .= 0.0
    cb.gradf .= 0.0
    cb.mvz .= 0.0
    cb.mvzdot .= 0.0
    cb.gradv .= 0.0
  end
  
  """
  Function that averages all the quantities by the number of trajectories
  """
  function average!(ntraj::Int64,cb::MCVBTCallback)
    cb.dkl /= ntraj
    cb.aval /= ntraj
    cb.gradf /= ntraj
    cb.gradv /= ntraj
  end
  
  """
  Implementation of the Monte Carlo Value Baseline algorithm
  """
  function callback(cb::MCVBTCallback,system::AbstractThermostattedSystem,mm::MixedModel,args...)
    # extract some arguments
    ind::Int64 = args[1]
    dt::Float64 = args[2]
  
    # compute gradient of variable force and update ydot
    jacobian!(cb.mvydot,system,mm.potentials[length(mm.potentials)])
    cb.mvydot .*= (sqrt(dt)/system.thermostat.scale)
    cb.mvy .+= (cb.mvydot*system.thermostat.rands)
  
    # calculate value baseline function
    vval::Float64 = callvbl(system,cb.vbl)
  
    # compute gradient of value baseline
    gradient!(cb.mvzdot,system,cb.vbl)
    cb.mvz .+= (dt .* cb.mvzdot)
  
    ### compute the instantaneous return
#    println(mm.potentials[end].condition(system))
    if mm.potentials[end].condition(system)
      # action difference - positive term (dx/dt - Ftot)^2
      cb.em .= ((system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands).^2
      # action difference - negative term (dx/dt - F0)^2
      cb.em .-= (mm.potentials[end].f .+ (system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands).^2
    else
#      println("hey")
      # action difference - positive term
      cb.em .= ((system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands .- mm.potentials[end].f).^2
      # action difference - negative term
      cb.em .-= ((system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands).^2
    end
    # compute return
    rval::Float64 = -sum(cb.em) / (2*system.thermostat.scale^2)
  
    # add return to KL divergence
    cb.dkl += rval*dt
  
    # observable bias
    aval::Float64 = cb.Afunc(system,mm,ind,dt)
    rval += aval# + Bfunc(system,mm,ind,dt)
    cb.aval += aval
  
    # update the gradients of the force parameters and the value baseline parameters
    cb.gradf .+= (rval .* cb.mvy) .- ((vval/dt) .* (cb.mvydot*system.thermostat.rands))
    cb.gradv .+= (rval .* cb.mvz) .- (vval .* cb.mvzdot)
  end
  
#  """
#  Implementation of the Monte Carlo Value Baseline algorithm for an active particle simulation
#  """
#  function callback(cb::MCVBCallback,system::AbstractActiveSystem,mm::MixedModel,args...)
#    # extract some arguments
#    ind::Int64 = args[1]
#    dt::Float64 = args[2]
#  
#    # compute gradient of variable force and update ydot
#    jacobian!(cb.mvydot,system,mm.potentials[length(mm.potentials)])
#    for i in 1:system.dim
#      @. @views cb.mvydot[:,i] .*= (sqrt(dt) / system.thermostat.scale[i])
#    end
#    cb.mvy .+= (cb.mvydot*system.thermostat.rands)
#  
#    # calculate value baseline function
#    vval::Float64 = callvbl(system,cb.vbl)
#  
#    # compute gradient of value baseline
#    gradient!(cb.mvzdot,system,cb.vbl)
#    cb.mvz .+= (dt .* cb.mvzdot)
#  
#    ### compute the instantaneous return
#    # action difference - positive term
#    cb.em .= ((system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands).^2
#    # action difference - negative term
#    cb.em .-= (mm.potentials[length(mm.potentials)].f .+ (system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands).^2
#    # compute return
#    rval::Float64 = sum(cb.em ./ (2 .* system.thermostat.scale.^2))
#  
#    # add return to KL divergence
#    cb.dkl += rval*dt
#  
#    # observable bias
#    aval::Float64 = cb.Afunc(system,mm,ind,dt)
#    rval += aval# + Bfunc(system,mm,ind,dt)
#    cb.aval += aval
#  
#    # update the gradients of the force parameters and the value baseline parameters
#    cb.gradf .+= (rval .* cb.mvy) .- ((vval/dt) .* (cb.mvydot*system.thermostat.rands))
#    cb.gradv .+= (rval .* cb.mvz) .- (vval .* cb.mvzdot)
#  
#  end