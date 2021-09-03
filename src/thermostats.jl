struct DummyThermostat <: AbstractThermostat; end

# TODO need to be able to thermostat DoFs individually!!!!
"""
Generic Langevin thermostat for a single particle
"""
mutable struct Langevin <: AbstractLangevin
  T::Float64
  gamma::Float64
  scale::Float64
  seed::UInt64
  rng::AbstractRNG
  rands::Vector{Float64}
  function Langevin(T::Float64,gamma::Float64,seed::UInt64)
    new(T,gamma,sqrt(2.0*T/gamma),seed,MersenneTwister(seed),[])
  end
end
Langevin(T::Float64,gamma::Float64) = Langevin(T,gamma,rand(UInt64))

function applythermostat!(dt::Float64,sys::AbstractThermostattedSystem,langevin::Langevin)
  sys.x .+= (sqrt(dt) * langevin.scale) .* langevin.rands
end

"""
Langevin thermostat for a single particle with different frictions per DoF
"""
mutable struct LangevinND <: AbstractLangevin
  T::Float64
  gamma::Vector{Float64}
  scale::Vector{Float64}
  seed::UInt64
  rng::AbstractRNG
  rands::Vector{Float64}
  function LangevinND(T::Float64,gamma::Vector{Float64},seed::UInt64)
    new(T,gamma,sqrt.(2*T ./ gamma),seed,MersenneTwister(seed),[])
  end
end
LangevinND(T::Float64,gamma::Vector{Float64}) = LangevinND(T,gamma,rand(UInt64))

function applythermostat!(dt::Float64,sys::AbstractThermostattedSystem,langevin::LangevinND)
  sys.x .+= (sqrt(dt) .* langevin.scale .* langevin.rands)
end

"""
Langevin thermostat for a single active particle
"""
mutable struct ActiveLangevin <: AbstractLangevin
  Tt::Float64
  gammat::Float64
  Tr::Float64
  gammar::Float64
  scale::Vector{Float64}
  seed::UInt64
  rng::AbstractRNG
  rands::Vector{Float64}
  function ActiveLangevin(dim::Int64,Tt::Float64,gammat::Float64,Tr::Float64,gammar::Float64,seed::UInt64)
    scale = zeros(Float64,dim)
    scale[1:dim] .= sqrt(2.0*Tt/gammat)
    scale[end] = sqrt(2.0*Tr/gammar)
    new(Tt,gammat,Tr,gammar,scale,seed,MersenneTwister(seed),[])
  end
end
ActiveLangevin(dim::Int64,Tt::Float64,gammat::Float64,Tr::Float64,gammar::Float64) = ActiveLangevin(dim,Tt,gammat,Tr,gammar,rand(UInt64))

function applythermostat!(dt::Float64,sys::AbstractActiveSystem,langevin::ActiveLangevin)
  @. @views sys.x[1:sys.dim-1] .+= (dt*sys.v0) .* [cos(sys.x[end]),sin(sys.x[end])]
  sys.x .+= (sqrt(dt) .* langevin.scale .* langevin.rands)
end

"""
Functions for access to the random number generator
"""
function getseed(langevin::AbstractLangevin)
  return langevin.seed
end

function setseed!(seed::Union{Int64,UInt64},langevin::AbstractLangevin)
  langevin.seed = convert(UInt64,seed)
  langevin.rng = MersenneTwister(langevin.seed)
end

"""
Various functions for generating noise
"""
function gennoises(nsteps::Int64,dim::Int64,rng::AbstractRNG)
  randn(rng,Float64,(nsteps,dim))
end

function gennoises!(nsteps::Int64,dim::Int64,langevin::AbstractLangevin)
  langevin.rands = randn(langevin.rng,Float64,(nsteps,dim))
end

#function gennoise!(dim::Int64,langevin::AbstractLangevin)
#  langevin.rands = randn(langevin.rng,Float64,dim)
#end

function gennoise!(dim::Int64,langevin::AbstractLangevin)
  r1::Float64 = rand(Flaot64)
  r2::Float64 = rand(Flaot64)
  theta::Float64 = 2.0*4.0*atan(1.0)*r1
  r::Float64 = sqrt(-2.0*log(r2))
  langevin.rands[1] = r*cos(theta)
  langevin.rands[2] = r*sin(theta)
end

"""
"""
function diffusion!(du::Array{Float64},langevin::Langevin)
  du .= 1.0
end
