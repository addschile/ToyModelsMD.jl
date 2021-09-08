abstract type AbstractGaussianModel <: AbstractSinglePotential end

### struct for defining a Gaussian force model
mutable struct GaussianModel <: AbstractGaussianModel
  dim::Int64
  nbasis::Int64
  shape::Tuple{Vararg{Int64}}
  mus::Vector{Vector{Float64}}
  sigs::Vector{Vector{Float64}}
  theta::Matrix{Float64}
  f::Vector{Float64}
  em::Vector{Float64}
  function GaussianModel(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},
                         theta::Matrix{Float64})
    dim = length(shape)
    nbasis = reduce(*,shape,init=1)
    new(dim,nbasis,shape,mus,sigs,theta,zeros(Float64,dim),zeros(Float64,nbasis))
  end
end

function GaussianModel(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}})
  dim::Int64 = length(shape)
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Matrix{Float64} = zeros(Float64, (nbasis,dim))
  GaussianModel(shape,mus,sigs,theta)
end

"""
struct for defining a time-dependent Gaussian force model
make sure the last set of basis functions is the time-dependence
"""
mutable struct GaussianModelTD <: AbstractGaussianModel
  dim::Int64
  nbasis::Int64
  shape::Tuple{Vararg{Int64}}
  mus::Vector{Vector{Float64}}
  sigs::Vector{Vector{Float64}}
  theta::Matrix{Float64}
  f::Vector{Float64}
  em::Vector{Float64}
  function GaussianModelTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},
                         theta::Matrix{Float64})
    dim = length(shape)-1
    nbasis = reduce(*,shape,init=1)
    new(dim,nbasis,shape,mus,sigs,theta,zeros(Float64,dim),zeros(Float64,nbasis))
  end
end

function GaussianModelTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}})
  dim::Int64 = length(shape)-1
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Matrix{Float64} = zeros(Float64, (nbasis,dim))
  GaussianModelTD(shape,mus,sigs,theta)
end

"""
Helper functions that get useful information about the model
"""
function getdimensionality(gm::AbstractGaussianModel)
  return gm.dim
end

# shape of the jacobian with respect to the parameters
function jacshape(gm::AbstractGaussianModel)
  return (gm.dim*gm.nbasis,gm.dim)
end

# TODO do i need this
function gradshape(gm::AbstractGaussianModel)
  return gm.dim*gm.nbasis
end

"""
Functions that compute the value of the basis functions
"""
function computebasis!(sys::AbstractSystem, gm::GaussianModel)
  k::Vector{Float64} = exp.( -0.5 .* (sys.x[1] .- gm.mus[1]).^2 ./ gm.sigs[1] )
  for i in 2:gm.dim
    k = vec( k * (exp.( -0.5 .* (sys.x[i] .- gm.mus[i]).^2 ./ gm.sigs[i] ))' )
  end
  gm.em .= k
end

function computebasis!(sys::AbstractSystem, gm::GaussianModelTD)
  k::Vector{Float64} = exp.( -0.5 .* (sys.x[1] .- gm.mus[1]).^2 ./ gm.sigs[1] )
  for i in 2:gm.dim
    k = vec( k * (exp.( -0.5 .* (sys.x[i] .- gm.mus[i]).^2 ./ gm.sigs[i] ))' )
  end
  k = vec( k * (exp.( -0.5 .* (sys.t .- gm.mus[end]).^2 ./ gm.sigs[end] ))' )
  gm.em .= k
end

"""
Force function with return value
"""
function force(sys::AbstractSystem, gm::AbstractGaussianModel; compbasis::Bool=true)
  if compbasis
    computebasis!(sys,gm)
  end
  fout::Vector{Float64} = zeros(Float64,gm.dim)
  for i in 1:gm.dim
    fout[i] = sum( @views gm.theta[:,i] .* gm.em )
  end
  return fout
end

"""
Force function with in-place changing
"""
function force!(sys::AbstractSystem, gm::AbstractGaussianModel; compbasis::Bool=true)
  if compbasis
    computebasis!(sys,gm)
  end
  for i in 1:gm.dim
    gm.f[i] = sum( @views gm.theta[:,i] .* gm.em )
  end
end

"""
Gradient wrt parameters for gaussian model
"""
function jacobian!(jac::Array{Float64},sys::AbstractSystem,gm::AbstractGaussianModel;
                   compbasis::Bool=false)
  if compbasis
    computebasis!(sys,gm)
  end
  ind0::Int64 = 1
  indf::Int64 = gm.nbasis
  for i in 1:gm.dim
    @. @views jac[ind0:indf,i] .= gm.em
    ind0 += gm.nbasis
    indf += gm.nbasis
  end
end

"""
Gradient wrt parameters for gaussian model
"""
function gradient!(grad::Array{Float64},sys::AbstractSystem,gm::AbstractGaussianModel;
                   compbasis::Bool=false)
  if compbasis
    computebasis!(sys,gm)
  end
  for i in 1:gm.dim
    grad[:,i] .= gm.em
  end
end

"""
Update parameters for the neural network model
"""
function updateparams!(lr::Float64,dtheta::Vector{Float64},gm::AbstractGaussianModel)
    # update coefficients for forces
    gm.theta .+= (lr .* reshape(dtheta, (gm.nbasis,gm.dim)))
end

"""
Set parameters for the neural network model
"""
function setparams!(theta::Vector{Float64},gm::AbstractGaussianModel)
    # update coefficients for forces
    gm.theta .= theta
end

function setparams!(gm::AbstractGaussianModel)
end


mutable struct GaussianModel2D <: AbstractSinglePotential
  dim::Int64
  nx::Int64
  ny::Int64
  nt::Int64
  mux::Vector{Float64}
  muy::Vector{Float64}
  mut::Vector{Float64}
  sigx::Vector{Float64}
  sigy::Vector{Float64}
  sigt::Vector{Float64}
  theta::Matrix{Float64}
  f::Vector{Float64}
  em::Vector{Float64}
  function GaussianModel2D(nx::Int64,ny::Int64,nt::Int64,
                           mux::Vector{Float64},muy::Vector{Float64},mut::Vector{Float64},
                           sigx::Vector{Float64},sigy::Vector{Float64},sigt::Vector{Float64},
                           theta::Matrix{Float64})
    new(2,nx,ny,nt,mux,muy,mut,sigx,sigy,sigt,theta,zeros(Float64,2),zeros(Float64,(nx*ny*nt)))
  end
end

function getdimensionality(gm::GaussianModel2D)
  return gm.dim
end

function jacshape(gm::GaussianModel2D)
  return (2*gm.nx*gm.ny*gm.nt,2)
end

function gradshape(gm::GaussianModel2D)
  return 2*gm.nx*gm.ny*gm.nt
end

"""
Force function with return value
"""
function force(sys::AbstractSystem, gm::GaussianModel2D)
  gm.em .= vec(kron(exp.(-0.5 .* (sys.t .- gm.mut).^2 ./ gm.sigt),
               kron(exp.(-0.5 .* (sys.x[2] .- gm.muy).^2 ./ gm.sigy),
                    exp.(-0.5 .* (sys.x[1] .- gm.mux).^2 ./ gm.sigx))))
  fout::Vector{Float64} = zeros(Float64,2)
  fout[1] = sum( @views gm.theta[:,1] .* gm.em )
  fout[2] = sum( @views gm.theta[:,2] .* gm.em )
  return fout
end

"""
Force function with in-place changing
"""
function force!(sys::AbstractSystem, gm::GaussianModel2D)
  gm.em .= vec(kron(exp.(-0.5 .* (sys.t .- gm.mut).^2 ./ gm.sigt),
               kron(exp.(-0.5 .* (sys.x[2] .- gm.muy).^2 ./ gm.sigy),
                    exp.(-0.5 .* (sys.x[1] .- gm.mux).^2 ./ gm.sigx))))
  gm.f[1] = sum( @views gm.theta[:,1] .* gm.em )
  gm.f[2] = sum( @views gm.theta[:,2] .* gm.em )
  #println(gm.f)
end

"""
Gradient wrt parameters for gaussian model
"""
function jacobian!(jac::Array{Float64},sys::AbstractSystem,gm::GaussianModel2D)
  @. @views jac[1:(gm.nx*gm.ny*gm.nt),1] .= gm.em
  @. @views jac[(gm.nx*gm.ny*gm.nt+1):(2*gm.nx*gm.ny*gm.nt),2] .= gm.em
end

"""
Gradient wrt parameters for gaussian model
"""
function gradient!(grad::Array{Float64},sys::AbstractSystem,gm::GaussianModel2D)
  for i in 1:gm.dim
    grad[:,i] .= gm.em
  end
  #grad[:,1] .= gm.em
  #grad[:,2] .= gm.em
end

"""
Update parameters for the neural network model
"""
function updateparams!(lr::Float64,dtheta::Vector{Float64},gm::GaussianModel2D)
    # update coefficients for forces
    gm.theta .+= (lr .* reshape(dtheta, (gm.nx*gm.ny*gm.nt,2)))
end

"""
Set parameters for the neural network model
"""
function setparams!(theta::Vector{Float64},gm::GaussianModel2D)
    # update coefficients for forces
    gm.theta .= theta
end

function setparams!(gm::GaussianModel2D)
end
