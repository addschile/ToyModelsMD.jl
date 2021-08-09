mutable struct DummySystem <: AbstractSystem; end

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

mutable struct ThermostattedSystem <: AbstractSystem
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
