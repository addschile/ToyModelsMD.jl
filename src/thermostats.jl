struct DummyThermostat <: AbstractThermostat; end

mutable struct Langevin <: AbstractLangevin
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

"""
"""
mutable struct ActiveLangevin <: AbstractLangevin
  Tt::Float64
  gammat::Float64
  Tr::Float64
  gammar::Float64
  scale::Vector{Float64}
  rng::AbstractRNG
  rands::Vector{Float64}
  function ActiveLangevin(dim::Int64,Tt::Float64,gammat::Float64,Tr::Float64,gammar::Float64,seed::UInt64)
    scale = ones(Float64,dim+1)
    scale[1:dim] .= sqrt(2.0*Tt/gammat)
    scale[end] = sqrt(2.0*Tr/gammar)
    new(Tt,gammat,Tr,gammar,scale,MersenneTwister(seed),[])
  end
end
ActiveLangevin(dim::Int64,Tt::Float64,gammat::Float64,Tr::Float64,gammar::Float64) = ActiveLangevin(dim,Tt,gammat,Tr,gammar,rand(UInt64))

"""
"""
function gennoises(nsteps::Int64,dim::Int64,rng::AbstractRNG)
  randn(rng,Float64,(nsteps,dim))
end

function gennoises!(nsteps::Int64,dim::Int64,langevin::AbstractLangevin)
  langevin.rands = randn(langevin.rng,Float64,(nsteps,dim))
end

function gennoise!(dim::Int64,langevin::AbstractLangevin)
  langevin.rands = randn(langevin.rng,Float64,dim)
  #langevin.rands = ones(Float64,dim)
end

"""
"""
function diffusion!(du::Array{Float64},langevin::Langevin)
  du .= 1.0
end
