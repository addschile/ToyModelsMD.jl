using Random

abstract type AbstractThermostat end

struct DummyThermostat <: AbstractThermostat; end

mutable struct Langevin <: AbstractThermostat
  T::Float64
  gamma::Float64
  rng::AbstractRNG
  function Langevin(T::Float64,gamma::Float64,seed::UInt64)
    new(T,gamma,MersenneTwister(seed))
  end
end
Langevin(T::Float64,gamma::Float64) = Langevin(T,gamma,rand(UInt64))

function diffusion!(du::AbstractArray,langevin::Langevin)
  du .= 1.0
end
