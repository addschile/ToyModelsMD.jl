using Flux:Dense,Chain
using Flux:destructure,params
using Zygote:Params,Grads
using Zygote:jacobian

### struct for defining the Muller-Brown potential
mutable struct NNModel <: AbstractSinglePotential
  dim::Int64
  pars::Params
  pinfo::Vector{Tuple{Int64, Vararg{Int64, N} where N}}
  nn::Union{Chain,Dense}
  re::Any
  theta::Vector
  f::Vector{Float64}
  function NNModel(dim::Int64,nn::Union{Chain,Dense})
    pars = params(nn)
    pinfo = [size(p_) for p_ in pars]
    th,re = destructure(nn)
    new(dim,pars,pinfo,nn,re,th,zeros(Float64,dim))
  end
end

function getdimensionality(nnm::NNModel)
  return nnm.dim
end

function jacshape(nnm::NNModel)
  return (size(nnm.theta)[1],nnm.dim)
end

function gradshape(nnm::NNModel)
  return size(nnm.theta)[1]
end

"""
Force function with return value
"""
function force(sys::AbstractSystem, nnm::NNModel)
  f::Vector{Float64} = nnm.nn(push!(sys.x,sys.t))
  pop!(sys.x)
  return f
end

"""
Force function with in-place changing
"""
function force!(sys::AbstractSystem, nnm::NNModel)
  nnm.f .= nnm.nn(push!(sys.x,sys.t))
  pop!(sys.x)
end

"""
Jacobian wrt parameters for neural network model
"""
function jacobian!(jac::Array{Float64},sys::AbstractSystem,nnm::NNModel)
  # compute gradients of output wrt parametes
  push!(sys.x,sys.t)
  grads::Grads = jacobian(() -> nnm.nn(sys.x), nnm.pars)
  pop!(sys.x)
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
Update parameters for the neural network model
"""
function updateparams!(lr::Float64,dtheta::Vector{Float64},nnm::NNModel)
    # update coefficients for forces
    nnm.theta .+= (lr .* dtheta)
    setparams!(nnm)
end

"""
Set parameters for the neural network model
"""
function setparams!(nnm::NNModel)
    # update coefficients for forces
    nnm.nn = nnm.re(nnm.theta)
    nnm.pars = params(nnm.nn)
end

function setparams!(theta::Vector{Float64},nnm::NNModel)
    # update coefficients for forces
    nnm.theta .= theta
    setparams!(nnm)
end

#"""
#Gradient wrt parameters for neural network model
#"""
#function gradient!(grad::Array{Float64},sys::AbstractSystem,nnm::NNModel)
#  grad = jacobian(() -> model(x), p)
#  pop!(sys.x)
#  val::Float64 = 0.0
#  for i in 1:gm.nx
#    val = exp(-0.5*(sys.x[1]-gm.mux[i])^2/gm.sigx[i])
#    @. @views grad[i,:,:] .= val
#  end
#  for i in 1:gm.ny
#    val = exp(-0.5*(sys.x[2]-gm.muy[i])^2/gm.sigy[i])
#    @. @views grad[:,i,:] .*= val
#  end
#  for i in 1:gm.nt
#    val = exp(-0.5*(sys.t-gm.mut[i])^2/gm.sigt[i])
#    @. @views grad[:,:,i] .*= val
#  end
#end
