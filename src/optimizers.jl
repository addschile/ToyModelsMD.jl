"""
"""
mutable struct GradientDescent <: AbstractOptimizer
  lr::Float64
  function GradientDescent(lr::Float64)
    new(lr)
  end
end

"""
"""
mutable struct ADAMOptimizer <: AbstractOptimizer
  beta1::Float64
  beta2::Float64
  function ADAMOptimizer(beta1::Float64,beta2::Float64)
    new(beta1,beta2)
  end
end