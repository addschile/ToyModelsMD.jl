struct DummyThermostat <: AbstractThermostat; end

mutable struct Langevin <: AbstractThermostat
  T::Float64
  gamma::Float64
  scale::Float64
  rng::AbstractRNG
  rands::Vector{Float64}
  function Langevin(T::Float64,gamma::Float64,seed::UInt64)
    new(T,gamma,sqrt(2.0*T/gamma),MersenneTwister(seed),[])
  end
end
Langevin(T::Float64,gamma::Float64) = Langevin(T,gamma,rand(UInt64))

function gennoises(nsteps::Int64,dim::Int64,rng::AbstractRNG)
  randn(rng,Float64,(nsteps,dim))
end

function gennoises!(nsteps::Int64,dim::Int64,langevin::Langevin)
  langevin.rands = randn(langevin.rng,Float64,(nsteps,dim))
end

function gennoise!(dim::Int64,langevin::Langevin)
  langevin.rands = randn(langevin.rng,Float64,dim)
end

function diffusion!(du::Array{Float64},langevin::Langevin)
  du .= 1.0
end
