"""
Struct for Active Brownian particle system
"""
mutable struct ActiveBrownianSystem <: AbstractActiveSystem
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
  f::Vector{Float64} = zeros(dim+1)
  ActiveBrownianSystem(dim+1,v0,0.0,x,f,thermostat)
end

# TODO idk yet, but this needs to be fixed to allow the different gammas
function addforce!(dt::Float64,sys::ActiveBrownianSystem,model::AbstractSinglePotential)
  @. @views sys.x[1:sys.dim-1] .+= dt .* model.f[1:sys.dim-1] ./ sys.thermostat.gammat
  if model.dim == sys.dim
    sys.x[end] += dt .* model.f[end] ./ sys.thermostat.gammar
  end
end
