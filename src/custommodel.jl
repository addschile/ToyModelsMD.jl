### struct for defining the Muller-Brown potential
mutable struct CustomModel <: AbstractSinglePotential
  dim::Int64
  pfunc::Union{Function,nothing}
  ffunc::Union{Function,nothing}
  f::Vector{Float64}
  function CustomModel(dim::Int64,pfunc::Union{Function,nothing},ffunc::Union{Function,nothing})
    new(dim,pfunc,ffunc2,zeros(Float64,dim))
  end
end
CustomModel(dim::Int64;pfunc::Union{Function,nothing}=nothing,ffunc::Union{Function,nothing}=nothing) = CustomModel(dim,pfunc,ffunc)

function getdimensionality(model::CustomModel)
  return model.dim
end

"""
Potential functions
"""
function potential(system::AbstractSystem, model::CustomModel)
  if model.pfunc != nothing
    return model.pfunc(system)
  end
end

"""
Force functions with return value
"""
function force(system::AbstractSystem, model::CustomModel)
  if model.ffunc != nothing
    return model.ffunc(system)
  else
    return zeros(Float64,model.dim)
  end
end

"""
Force functions with in-place changing
"""
function force!(system::AbstractSystem, model::CustomModel)
  if model.ffunc != nothing
    model.ffunc(system,model.f)
  else
    modelf .= 0.0
  end
end
