mutable struct MCVBCallback <: AbstractCallback
  every::Int64
  dkl::Float64
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
  function MCVBCallback(Afunc::Function,model::AbstractPotential,vbl::AbstractValueBaseline)
    mvy::Array{Float64} = zeros(Float64,gradshape(model))
    mvydot::Array{Float64} = zeros(Float64,jacshape(model))
    gradf::Array{Float64} = zeros(Float64,gradshape(model))
    mvz::Array{Float64} = zeros(Float64,gradshape(vbl))
    mvzdot::Array{Float64} = zeros(Float64,gradshape(vbl))
    gradv::Array{Float64} = zeros(Float64,gradshape(vbl))
    em = zeros(Float64,getdimensionality(model))
    new(1,0.0,Afunc,mvy,mvydot,gradf,mvz,mvzdot,gradv,vbl,em)
  end
end

"""
Function that zeros all the quantities for restarting trajectories
"""
function initialize!(cb::MCVBCallback)
  cb.dkl = 0.0
  cb.mvy .= 0.0
  cb.mvydot .= 0.0
  cb.gradf .= 0.0
  cb.mvz .= 0.0
  cb.mvzdot .= 0.0
  cb.gradv .= 0.0
end

"""
Implementation of the Monte Carlo Value Baseline algorithm
"""
function callback(cb::MCVBCallback,system::AbstractSystem,mm::MixedModel,args...)
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
  # action difference - positive term
  cb.em .= ((system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands).^2
  # action difference - negative term
  cb.em .-= (mm.potentials[length(mm.potentials)].f .+ (system.thermostat.scale/sqrt(dt)) .* system.thermostat.rands).^2
  # compute return
  rval::Float64 = sum(cb.em) / (2*system.thermostat.scale^2)

  # add return to KL divergence
  cb.dkl += rval*dt

  # observable bias
  rval += cb.Afunc(system,mm,ind,dt)# + Bfunc(system,mm,ind,dt)

  # update the gradients of the force parameters and the value baseline parameters
  cb.gradf .+= (rval .* cb.mvy) .- ((vval/dt) .* (cb.mvydot*system.thermostat.rands))
  cb.gradv .+= (rval .* cb.mvz) .- (vval .* cb.mvzdot)
end
