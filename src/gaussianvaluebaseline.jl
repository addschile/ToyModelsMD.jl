abstract type AbstractGaussianValueBaseline <: AbstractValueBaseline end

### struct for defining a Gaussian force model
mutable struct GaussianValueBaseline <: AbstractGaussianValueBaseline
  dim::Int64
  nbasis::Int64
  shape::Tuple{Vararg{Int64}}
  mus::Vector{Vector{Float64}}
  sigs::Vector{Vector{Float64}}
  theta::Vector{Float64}
  em::Vector{Float64}
  function GaussianValueBaseline(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},
                         theta::Vector{Float64})
    dim = length(shape)
    nbasis = reduce(*,shape,init=1)
    new(dim,nbasis,shape,mus,sigs,theta,zeros(Float64,nbasis))
  end
end

function GaussianValueBaseline(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}})
  dim::Int64 = length(shape)
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Vector{Float64} = zeros(Float64,nbasis)
  GaussianValueBaseline(shape,mus,sigs,theta)
end

"""
struct for defining a time-dependent Gaussian force model
make sure the last set of basis functions is the time-dependence
"""
mutable struct GaussianValueBaselineTD <: AbstractGaussianValueBaseline
  dim::Int64
  nbasis::Int64
  shape::Tuple{Vararg{Int64}}
  mus::Vector{Vector{Float64}}
  sigs::Vector{Vector{Float64}}
  theta::Vector{Float64}
  em::Vector{Float64}
  function GaussianValueBaselineTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},
                         theta::Vector{Float64})
    dim = length(shape)-1
    nbasis = reduce(*,shape,init=1)
    new(dim,nbasis,shape,mus,sigs,theta,zeros(Float64,nbasis))
  end
end

function GaussianValueBaselineTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}})
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Vector{Float64} = zeros(Float64, nbasis)
  GaussianValueBaselineTD(shape,mus,sigs,theta)
end

"""
Helper functions that get useful information about the model
"""
function getdimensionality(gm::AbstractGaussianValueBaseline)
  return gm.dim
end

function gradshape(gm::AbstractGaussianValueBaseline)
  return gm.nbasis
end

"""
Functions that compute the value of the basis functions
"""
function computebasis!(sys::AbstractSystem, gm::GaussianValueBaseline)
  k::Vector{Float64} = exp.( -0.5 .* (sys.x[1] .- gm.mus[1]).^2 ./ gm.sigs[1] )
  for i in 2:gm.dim
    k = vec( k * (exp.( -0.5 .* (sys.x[i] .- gm.mus[i]).^2 ./ gm.sigs[i] ))' )
  end
  gm.em .= k
end

function computebasis!(sys::AbstractSystem, gm::GaussianValueBaselineTD)
  k::Vector{Float64} = exp.( -0.5 .* (sys.x[1] .- gm.mus[1]).^2 ./ gm.sigs[1] )
  for i in 2:gm.dim
    k = vec( k * (exp.( -0.5 .* (sys.x[i] .- gm.mus[i]).^2 ./ gm.sigs[i] ))' )
  end
  k = vec( k * (exp.( -0.5 .* (sys.t .- gm.mus[end]).^2 ./ gm.sigs[end] ))' )
  gm.em .= k
end

"""
Function that evaluates the gaussian value baseline function
"""
function callvbl(sys::AbstractSystem,vbl::AbstractGaussianValueBaseline; compbasis::Bool=true)
  if compbasis
    computebasis!(sys,vbl)
  end
  return sum( vbl.theta .* vbl.em )
end


"""
Gradient wrt parameters for gaussian model
"""
function gradient!(grad::Array{Float64},sys::AbstractSystem,gm::AbstractGaussianValueBaseline;
                   compbasis::Bool=false)
  if compbasis
    computebasis!(sys,gm)
  end
  grad .= gm.em
end

"""
Update parameters for the neural network model
"""
function updateparams!(lr::Float64,dtheta::Vector{Float64},gm::AbstractGaussianValueBaseline)
    # update coefficients for forces
    gm.theta .+= lr .* dtheta
end

"""
Set parameters for the neural network model
"""
function setparams!(theta::Vector{Float64},gm::AbstractGaussianValueBaseline)
    # update coefficients for forces
    gm.theta .= theta
end

function setparams!(gm::GaussianModel)
end

mutable struct GaussianValueBaseline2D <: AbstractValueBaseline
  nx::Int64
  ny::Int64
  nt::Int64
  mux::Vector{Float64}
  muy::Vector{Float64}
  mut::Vector{Float64}
  sigx::Vector{Float64}
  sigy::Vector{Float64}
  sigt::Vector{Float64}
  theta::Vector{Float64}
  em::Vector{Float64}
  function GaussianValueBaseline2D(nx::Int64,ny::Int64,nt::Int64,
                                   mux::Vector{Float64},muy::Vector{Float64},mut::Vector{Float64},
                                   sigx::Vector{Float64},sigy::Vector{Float64},sigt::Vector{Float64},
                                   theta::Vector{Float64})
    new(nx,ny,nt,mux,muy,mut,sigx,sigy,sigt,theta,zeros(Float64,size(theta)))
  end
end

function gradshape(vbl::GaussianValueBaseline2D)
  return size(vbl.theta)
end

function callvbl(sys::AbstractSystem,vbl::GaussianValueBaseline2D)
  vbl.em .= vec(kron(exp.(-0.5 .* (sys.t .- vbl.mut).^2 ./ vbl.sigt),
                kron(exp.(-0.5 .* (sys.x[2] .- vbl.muy).^2 ./ vbl.sigy),
                     exp.(-0.5 .* (sys.x[1] .- vbl.mux).^2 ./ vbl.sigx))))
  return sum( vbl.theta.*vbl.em )
end

"""
Gradient wrt parameters for gaussian value baseline
"""
function gradient!(grad::Array{Float64},sys::AbstractSystem,vbl::GaussianValueBaseline2D)
  grad .= vbl.em
end

"""
Update parameters for the neural network model
"""
function updateparams!(lr::Float64,dtheta::Vector{Float64},vbl::GaussianValueBaseline2D)
    # update coefficients for forces
    vbl.theta .+= (lr .* dtheta)
end

"""
Set parameters for the neural network model
"""
function setparams!(theta::Vector{Float64},vbl::GaussianValueBaseline2D)
    # update coefficients for forces
    vbl.theta .= theta
end

function setparams!(gm::GaussianValueBaseline2D)
end