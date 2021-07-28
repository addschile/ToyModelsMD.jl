abstract type AbstractSystem end

mutable struct DummySystem <: AbstractSystem; end

mutable struct ThermostattedSystem <: AbstractSystem
  model::AbstractPotential
  thermostat::AbstractThermostat
  x::Array{Float64}
  f::Array{Float64}
  em::AbstractArray
  function ThermostattedSystem(model::AbstractPotential,thermostat::AbstractThermostat)
    dim::Int64 = getdimensionality(model)
    x::Array{Float64} = zeros(dim)
    f::Array{Float64} = zeros(dim)
    em::AbstractArray = genextramem(model)
    new(model,thermostat,x,f,em)
  end
end

function fderivative(du::AbstractArray,u::AbstractArray,sys::AbstractSystem)
  du .*= 0.0
  force!(u,du,sys.em,sys.model)
end

function gderivative(du::AbstractArray,u::AbstractArray,therm::AbstractThermostat)
  du .*= 0.0
  diffusion!(du,therm)
end
