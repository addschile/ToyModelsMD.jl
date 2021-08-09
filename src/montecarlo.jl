abstract type AbstractMCSampler end

mutable struct MCSampler <: AbstractMCSampler
  npersweep::Int64
  beta::Float64
  stepsize::Float64
  energy::Float64
  rng::AbstractRNG
  system::System
  dx::Array{Float64}
  function MCSampler(npersweep::Int64,beta::Float64,system::System,seed::UInt64)
    new(npersweep,beta,MersenneTwister(seed),system,zeros(Float64,size(system.x)))
  end
end
MCSampler(npersweep::Int64,beta::Float64,system::System) = MCSampler(npersweep,beta,system,rand(UInt64))
MCSampler(beta::Float64,system::System) = MCSampler(1,beta,system,rand(UInt64))
MCSampler(system::System,seed::UInt64) = MCSampler(1,1.0,system,seed)
MCSampler(system::System) = MCSampler(1,1.0,system,rand(UInt64))

mutable struct BiasedMCSampler <: AbstractMCSampler
  npersweep::Int64
  beta::Float64
  stepsize::Float64
  energy::Float64
  rng::AbstractRNG
  system::System
  bias::Function
  dx::Array{Float64}
  function BiasedMCSampler(npersweep::Int64,beta::Float64,system::System,bias::Function,seed::UInt64)
    new(npersweep,beta,MersenneTwister(seed),system,bias,zeros(Float64,size(system.x)))
  end
end
BiasedMCSampler(npersweep::Int64,beta::Float64,system::System,bias::Function) = MCSampler(npersweep,beta,system,bias,rand(UInt64))
BiasedMCSampler(beta::Float64,system::System,bias::Function) = MCSampler(1,beta,system,bias,rand(UInt64))
BiasedMCSampler(system::System,bias::Function,seed::UInt64) = MCSampler(1,1.0,system,bias,seed)
BiasedMCSampler(system::System,bias::Function) = MCSampler(1,1.0,system,bias,rand(UInt64))

function runsweeps!(nsweeps::Int64,mc::AbstractMCSampler)
  for i in 1:nsweeps
    sweep!(mc)
  end
end

function runsweeps!(nsweeps::Int64,mc::AbstractMCSampler,cb::AbstractCallback)
  for i in 1:nsweeps
    if i%cb.every==0
      callback(i,mc.system,cb)
    end
    sweep!(mc)
  end
end

function sweep!(mc::AbstractMCSampler)
  rands = mc.stepsize*gennoises(mc.npersweep,getdimensionality(mc.system.model),mc.rng)
  for i in 1:mc.npersweep
    @. @views mc.dx .= rands[i,:]
    step!(mc)
  end
end

function step!(mc::MCSampler)
  dE::Float64 = potential(mc.system.x .+ dx, mc.system.model) - potential(mc.system.x, mc.system.model)
  if dE < 0.
    mc.x .+= mc.dx
  else rand(rng) < exp(-beta*dE)
    mc.x .+= mc.dx
  end
end

function step!(mc::BiasedMCSampler)
  dE::Float64 = potential(mc.system.x .+ dx, mc.system.model) - potential(mc.system.x, mc.system.model)
  dE += mc.bias(mc.system.x .+ dx) - mc.bias(mc.system.x)
  if dE < 0.
    mc.x .+= mc.dx
  else rand(rng) < exp(-beta*dE)
    mc.x .+= mc.dx
  end
end
