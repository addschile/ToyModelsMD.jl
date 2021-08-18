### struct for defining the Muller-Brown potential
mutable struct CustomTrainableModel <: AbstractSinglePotential
  dim::Int64
  nparams::Int64
  theta::Vector{Float64}
  pfunc::Union{Function,nothing}
  ffunc::Union{Function,nothing}
  f::Vector{Float64}
  function CustomTrainableModel(dim::Int64,pfunc::Union{Function,nothing},ffunc::Union{Function,nothing})
    new(dim,pfunc,ffunc2,zeros(Float64,dim))
  end
end
CustomTrainableModel(dim::Int64;pfunc::Union{Function,nothing}=nothing,ffunc::Union{Function,nothing}=nothing) = CustomTrainableModel(dim,pfunc,ffunc)

function getdimensionality(model::CustomTrainableModel)
  return model.dim
end

function jacshape(model::CustomTrainableModel)
  return (model.nparams,model.dim)
end

function gradshape(model::CustomTrainableModel)
  return model.nparams
end

"""
Potential functions
"""
function potential(system::AbstractSystem, model::CustomTrainableModel)
  if model.pfunc != nothing
    return model.pfunc(system)
  end
end

"""
Force functions with return value
"""
function force(system::AbstractSystem, model::CustomTrainableModel)
  if model.ffunc != nothing
    return model.ffunc(system)
  else
    return zeros(Float64,model.dim)
  end
end

"""
Force functions with in-place changing
"""
function force!(system::AbstractSystem, model::CustomTrainableModel)
  if model.ffunc != nothing
    model.ffunc(system,model.f)
  else
    modelf .= 0.0
  end
end

"""
Jacobian wrt parameters for neural network model
"""
function jacobian!(jac::Array{Float64},sys::AbstractSystem,model::CustomTrainableModel)
  # TODO
  grads::Grads = jacobian(() -> nnm.ffunc(sys), nnm.pars)
  # store gradients into jacobian array
  bcount::Int64 = 1
  for i in 1:length(nnm.pars)
    # TODO replace with reduce function
    ptup::Tuple = nnm.pinfo[i]
    ecount::Int64 = 1
    for j in 1:length(ptup)
      ecount *= ptup[j]
    end
    ecount += (bcount - 1)
    @. @views jac[bcount:ecount,:] .= grads[nnm.pars[i]]'
    bcount = ecount + 1
  end
end

"""
Update parameters for the model
"""
function updateparams!(lr::Float64,dtheta::Vector{Float64},model::CustomTrainableModel)
    # update coefficients for forces
    model.theta .+= (lr .* dtheta)
end

"""
Set parameters for the model
"""
function setparams!(theta::Vector{Float64},model::CustomTrainableModel)
    # update coefficients for forces
    model.theta .= theta
end

