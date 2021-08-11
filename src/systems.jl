struct DummySystem <: AbstractSystem; end

mutable struct System <: AbstractSystem
  dim::Int64
  t::Float64
  x::Vector{Float64}
  f::Vector{Float64}
  function System(dim::Int64,t::Float64,x::Vector{Float64},f::Vector{Float64})
    new(dim,t,x,f)
  end
end

function System(model::AbstractPotential)
  dim::Int64 = getdimensionality(model)
  x::Vector{Float64} = zeros(dim)
  f::Vector{Float64} = zeros(dim)
  System(dim,0.0,x,f)
end

"""
"""
mutable struct ThermostattedSystem <: AbstractThermostattedSystem
  dim::Int64
  t::Float64
  x::Vector{Float64}
  f::Vector{Float64}
  thermostat::AbstractThermostat
  function ThermostattedSystem(dim::Int64,t::Float64,x::Vector{Float64},f::Vector{Float64},thermostat::AbstractThermostat)
    new(dim,t,x,f,thermostat)
  end
end

function ThermostattedSystem(model::AbstractPotential,thermostat::AbstractThermostat)
  dim::Int64 = getdimensionality(model)
  x::Vector{Float64} = zeros(dim)
  f::Vector{Float64} = zeros(dim)
  ThermostattedSystem(dim,0.0,x,f,thermostat)
end

function zeroforce!(system::AbstractSystem)
  system.f .= 0.0
end

function addforce!(dt::Float64,system::AbstractSystem,model::AbstractPotential)
  system.x .+= dt .* model.f
end

"""
"""
mutable struct ActiveBrownianSystem <: AbstractThermostattedSystem
  dim::Int64
  v0::Float64
  t::Float64
  x::Vector{Float64}
  f::Vector{Float64}
  thermostat::ActiveLangevin
  function ActiveBrownianSystem(dim::Int64,v0::Float64,t::Float64,x::Vector{Float64},f::Vector{Float64},thermostat::ActiveLangevin)
    new(dim,v0,t,x,f,thermostat)
  end
end

function ActiveBrownianSystem(v0::Float64,model::AbstractPotential,thermostat::ActiveLangevin)
  dim::Int64 = getdimensionality(model)
  x::Vector{Float64} = zeros(dim+1)
  f::Vector{Float64} = zeros(dim)
  ActiveBrownianSystem(dim+1,v0,0.0,x,f,thermostat)
end

function addforce!(dt::Float64,sys::ActiveBrownianSystem,model::AbstractPotential)
  @. @views sys.x[1:sys.dim-1] .+= dt .* (model.f + (sys.v0 .* [cos(sys.x[end]),sin(sys.x[end])]))
end

#function fderivative(du::AbstractArray,u::AbstractArray,sys::AbstractSystem)
#  force!(u,du,sys.em,sys.model)
#end
#
#function gderivative(du::AbstractArray,u::AbstractArray,therm::AbstractThermostat)
#  diffusion!(du,therm)
#end
#
#mutable struct OverdampedSystem <: AbstractSystem
#  model::AbstractPotential
#  thermostat::AbstractThermostat
#  x::Array{Float64}
#  f::Array{Float64}
#  em::AbstractArray
#  function OverdampedSystem(model::AbstractPotential,thermostat::AbstractThermostat)
#    dim::Int64 = getdimensionality(model)
#    x::Array{Float64} = zeros(dim)
#    f::Array{Float64} = zeros(dim)
#    em::AbstractArray = genextramem(model)
#    new(model,thermostat,x,f,em)
#  end
#end
