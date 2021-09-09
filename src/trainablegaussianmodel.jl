abstract type AbstractTrainableGaussianModel <: AbstractTrainablePotential end

### struct for defining a Gaussian force model
mutable struct TrainableGaussianModel <: AbstractTrainableGaussianModel
  dim::Int64
  nbasis::Int64
  shape::Tuple{Vararg{Int64}}
  mus::Vector{Vector{Float64}}
  sigs::Vector{Vector{Float64}}
  theta::Matrix{Float64}
  condition::Function
  f::Vector{Float64}
  em::Vector{Float64}
  function TrainableGaussianModel(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},
                                  theta::Matrix{Float64},condition::Function)
    dim = length(shape)
    nbasis = reduce(*,shape,init=1)
    new(dim,nbasis,shape,mus,sigs,theta,condition,zeros(Float64,dim),zeros(Float64,nbasis))
  end
end

"""
Constructor with zeros for parameters and default true condition
"""
function TrainableGaussianModel(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}})
  dim::Int64 = length(shape)
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Matrix{Float64} = zeros(Float64, (nbasis,dim))
  condition::Fucntion = defaultcondition
  TrainableGaussianModel(shape,mus,sigs,theta,condition)
end

"""
Constructor with zeros for parameters
"""
function TrainableGaussianModel(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},condition::Function)
  dim::Int64 = length(shape)
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Matrix{Float64} = zeros(Float64, (nbasis,dim))
  TrainableGaussianModel(shape,mus,sigs,theta,condition)
end

"""
Constructor with default true condition
"""
function TrainableGaussianModel(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},theta::Matrix{Float64})
  dim::Int64 = length(shape)
  nbasis::Int64 = reduce(*,shape,init=1)
  condition::Function = defaultcondition
  TrainableGaussianModel(shape,mus,sigs,theta,condition)
end

"""
struct for defining a time-dependent Gaussian force model
make sure the last set of basis functions is the time-dependence
"""
mutable struct TrainableGaussianModelTD <: AbstractTrainableGaussianModel
  dim::Int64
  nbasis::Int64
  shape::Tuple{Vararg{Int64}}
  mus::Vector{Vector{Float64}}
  sigs::Vector{Vector{Float64}}
  theta::Matrix{Float64}
  condition::Function
  f::Vector{Float64}
  em::Vector{Float64}
  function TrainableGaussianModelTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},
                         theta::Matrix{Float64},condition::Function)
    dim = length(shape)-1
    nbasis = reduce(*,shape,init=1)
    new(dim,nbasis,shape,mus,sigs,theta,condition,zeros(Float64,dim),zeros(Float64,nbasis))
  end
end

"""
Constructor with zeros for parameters and default true condition
"""
function TrainableGaussianModelTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}})
  dim::Int64 = length(shape)-1
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Matrix{Float64} = zeros(Float64, (nbasis,dim))
  condition::Function = defaultcondition
  TrainableGaussianModelTD(shape,mus,sigs,theta,condition)
end

"""
Constructor with zeros for parameters
"""
function TrainableGaussianModelTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},condition::Function)
  dim::Int64 = length(shape)-1
  nbasis::Int64 = reduce(*,shape,init=1)
  theta::Matrix{Float64} = zeros(Float64, (nbasis,dim))
  TrainableGaussianModelTD(shape,mus,sigs,theta,condition)
end

"""
Constructor with default true condition
"""
function TrainableGaussianModelTD(shape::Tuple{Vararg{Int64}},mus::Vector{Vector{Float64}},sigs::Vector{Vector{Float64}},theta::Matrix{Float64})
  dim::Int64 = length(shape)-1
  nbasis::Int64 = reduce(*,shape,init=1)
  condition::Function = defaultcondition
  TrainableGaussianModelTD(shape,mus,sigs,theta,condition)
end


"""
Helper functions that get useful information about the model
"""
function getdimensionality(gm::AbstractTrainableGaussianModel)
  return gm.dim
end

# shape of the jacobian with respect to the parameters
function jacshape(gm::AbstractTrainableGaussianModel)
  return (gm.dim*gm.nbasis,gm.dim)
end

# TODO do i need this
function gradshape(gm::AbstractTrainableGaussianModel)
  return gm.dim*gm.nbasis
end

"""
Default condition on whether to use model in equation of motion
"""
function defaultcondition(sys::AbstractSystem)
  return true
end

"""
Functions that compute the value of the basis functions
"""
function computebasis!(sys::AbstractSystem, gm::TrainableGaussianModel)
  k::Vector{Float64} = exp.( -0.5 .* (sys.x[1] .- gm.mus[1]).^2 ./ gm.sigs[1] )
  for i in 2:gm.dim
    k = vec( k * (exp.( -0.5 .* (sys.x[i] .- gm.mus[i]).^2 ./ gm.sigs[i] ))' )
  end
  gm.em .= k
end

function computebasis!(sys::AbstractSystem, gm::TrainableGaussianModelTD)
  # forward
  k::Vector{Float64} = exp.( -0.5 .* (sys.x[1] .- gm.mus[1]).^2 ./ gm.sigs[1] )
  for i in 2:gm.dim
    k = vec( k * (exp.( -0.5 .* (sys.x[i] .- gm.mus[i]).^2 ./ gm.sigs[i] ))' )
  end
  println(sys.t)
  k = vec( k * (exp.( -0.5 .* (sys.t .- gm.mus[end]).^2 ./ gm.sigs[end] ))' )
  gm.em .= k
end

"""
Force function with return value
"""
function force(sys::AbstractSystem, gm::AbstractTrainableGaussianModel; compbasis::Bool=true)
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
function force!(sys::AbstractSystem, gm::AbstractTrainableGaussianModel; compbasis::Bool=true)
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
function jacobian!(jac::Array{Float64},sys::AbstractSystem,gm::AbstractTrainableGaussianModel;
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
function gradient!(grad::Array{Float64},sys::AbstractSystem,gm::AbstractTrainableGaussianModel;
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
function updateparams!(lr::Float64,dtheta::Vector{Float64},gm::AbstractTrainableGaussianModel)
    # update coefficients for forces
    gm.theta .+= (lr .* reshape(dtheta, (gm.nbasis,gm.dim)))
end

"""
Set parameters for the neural network model
"""
function setparams!(theta::Vector{Float64},gm::AbstractTrainableGaussianModel)
    # update coefficients for forces
    gm.theta .= theta
end

function setparams!(gm::AbstractTrainableGaussianModel)
end