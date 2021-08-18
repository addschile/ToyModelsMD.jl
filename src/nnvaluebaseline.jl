using Flux:Dense,Chain
using Flux:destructure,params
using Zygote:Params,Grads
using Zygote:gradient

mutable struct NNValueBaseline <: AbstractValueBaseline
  pars::Params
  pinfo::Vector{Tuple{Int64, Vararg{Int64, N} where N}}
  nn::Union{Chain,Dense}
  re::Any
  theta::Vector
  function NNValueBaseline(nn::Union{Chain,Dense})
    pars = params(nn)
    pinfo = [size(p_) for p_ in pars]
    theta,re = destructure(nn)
    new(pars,pinfo,nn,re,theta)
  end
end

function gradshape(vbl::NNValueBaseline)
  return size(vbl.theta)[1]
end

function callvbl(sys::AbstractSystem,vbl::NNValueBaseline)
  vout::Float64 = vbl.nn(push!(sys.x,sys.t))[1]
  pop!(sys.x)
  return vout
end

"""
Gradient wrt parameters for gaussian value baseline
"""
function gradient!(grad::Vector{Float64},sys::AbstractSystem,vbl::NNValueBaseline)
  # compute gradients of output wrt parametes
  push!(sys.x,sys.t)
  grads::Grads = gradient(() -> vbl.nn(sys.x)[1], vbl.pars)
  pop!(sys.x)
  # store gradients into jacobian array
  bcount::Int64 = 1
  for i in 1:length(vbl.pars)
    # TODO replace with reduce function
    ptup::Tuple = vbl.pinfo[i]
    ecount::Int64 = 1
    for j in 1:length(ptup)
      ecount *= ptup[j]
    end
    ecount += (bcount - 1)
    @views grad[bcount:ecount] .= vec(grads[vbl.pars[i]])
    bcount = ecount + 1
  end
end

"""
Update parameters for the neural network value baseline
"""
function updateparams!(lr::Float64,dtheta::Vector{Float64},vbl::NNValueBaseline)
    # update coefficients for forces
    vbl.theta .+= (lr .* dtheta)
    setparams!(vbl)
end

"""
Set parameters for the neural network model
"""
function setparams!(vbl::NNValueBaseline)
    # update coefficients for forces
    vbl.nn = vbl.re(vbl.theta)
    vbl.pars = params(vbl.nn)
end

function setparams!(theta::Vector{Float64},vbl::NNModel)
    # update coefficients for forces
    vbl.theta .= theta
    setparams!(vbl)
end
