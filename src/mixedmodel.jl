### struct for defining a mixed-model potential
mutable struct MixedModel <: AbstractMixedModel
  dim::Int64
  potentials::Vector{AbstractPotential}
  function MixedModel(dim::Int64,potentials::Vector{AbstractPotential})
    new(dim,potentials)
  end
end

function getdimensionality(mm::MixedModel)
  return mm.dim
end

"""
Force functions with return value
"""
function force(system::AbstractSystem, mm::MixedModel)
  fout::Array{Float64} = zeros(Float64,size(system.x))
  for pot in mm.potentials
    fout .+= force(system,pot)
  end
  return fout
end

"""
Force functions with in-place changing
"""
function force!(system::AbstractSystem, mm::MixedModel)
  for pot in mm.potentials
    force!(system,pot)
  end
end

"""
Force function with in-place changing of a particular model
"""
function force!(ind::Int64,system::AbstractSystem,mm::MixedModel)
  force!(system,mm.potentials[ind])
end

function addforce!(dt::Float64,system::AbstractSystem,mm::MixedModel)
  for pot in mm.potentials
    addforce!(dt,system,pot)
  end
end

function addforce!(dt::Float64,sys::ActiveBrownianSystem,mm::MixedModel)
  for pot in mm.potentials
    if pot.dim == sys.dim
      sys.x += dt .* pot.f
    else
      @. @views sys.x[1:sys.dim-1] .+= dt .* pot.f[1:sys.dim-1]
    end
  end
  @. @views sys.x[1:sys.dim-1] .+= (dt*sys.v0) .* [cos(sys.x[end]),sin(sys.x[end])]
end

"""
Compute the gradient of the parameters for a particular model
"""
function gradient!(ind::Int64,grad::Array{Float64},system::AbstractSystem,mm::MixedModel)
  gradient!(grad,system,mm.potentials[ind])
end
