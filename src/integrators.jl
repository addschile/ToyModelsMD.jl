"""
Forward Euler integration scheme
"""
mutable struct Euler <: AbstractIntegrator
  system::System
  model::AbstractPotential
  function Euler(system::System,model::AbstractPotential)
    new(system,model)
  end
end

function runtraj!(nsteps::Int64,dt::Float64,int::Euler)
  for i in 1:nsteps
    step!(i,dt,int)
  end
end

function runtraj!(nsteps::Int64,dt::Float64,int::Euler,cb::AbstractCallback)
  force!(int.system,int.model)
  for i in 1:nsteps
    if i%cb.every==0
      callback(cb,int.system,int.model,i,dt)
    end
    step!(i,dt,int)
  end
end

function step!(ind::Int64,dt::Float64,int::Euler)
  # forward euler integrator
  # drift
  force!(int.system,int.model)
  addforce!(dt,int.system,int.model)
  int.system.t += dt
end

"""
Stochastic Euler integration scheme
"""
mutable struct StochasticEuler <: AbstractIntegrator
  system::AbstractThermostattedSystem
  model::AbstractPotential
  function StochasticEuler(system::AbstractThermostattedSystem,model::AbstractPotential)
    new(system,model)
  end
end

function runtraj!(nsteps::Int64,dt::Float64,int::StochasticEuler)
  for i in 1:nsteps
    step!(i,dt,int)
  end
end

function step!(ind::Int64,dt::Float64,int::StochasticEuler)
  # forward euler-murayama integrator
  # compute force for drift
  force!(int.system,int.model)
  # generate noise for diffusion
  gennoise!(int.system.dim,int.system.thermostat)
  # update with drift
  addforce!(dt,int.system,int.model)
  # update with diffusion
  int.system.x .+= sqrt(dt)*int.system.thermostat.scale .* int.system.thermostat.rands
  # update time
  int.system.t += dt
end

function runtraj!(nsteps::Int64,dt::Float64,int::StochasticEuler,cb::AbstractCallback)
  for i in 1:nsteps
    step!(i,dt,int,cb)
  end
end

function step!(ind::Int64,dt::Float64,int::StochasticEuler,cb::AbstractCallback)
  # forward euler-murayama integrator
  # compute force for drift
  force!(int.system,int.model)
  # generate noise for diffusion
  gennoise!(int.system.dim,int.system.thermostat)
  # callback
  if ind%cb.every==0
    callback(cb,int.system,int.model,ind,dt)
  end
  # update with drift
  addforce!(dt,int.system,int.model)
  # update with diffusion
  int.system.x .+= (sqrt(dt) .* int.system.thermostat.scale .* int.system.thermostat.rands)
  # update time
  int.system.t += dt
end
