### struct for defining the Muller-Brown potential
mutable struct MullerBrown <: AbstractSinglePotential
  dim::Int64
  A::Vector{Float64}
  a::Vector{Float64}
  b::Vector{Float64}
  c::Vector{Float64}
  x0s::Vector{Float64}
  y0s::Vector{Float64}
  f::Vector{Float64}
  em::Vector{Float64}
  function MullerBrown()
    new(2,[-200.,-100.,-170.,15.],[-1.,-1.,-6.5,0.7],[0.,0.,11.,0.6],[-10.,-10.,-6.5,0.7],[1.,0.,-0.5,-1.],[0.,0.5,1.5,1.],zeros(Float64,2),zeros(Float64,12))
  end
end

function getdimensionality(mb::MullerBrown)
  return 2
end

"""
Potential functions
"""
function potential(x::Vector{Float64}, mb::MullerBrown)
  @. @views mb.em[1:4] = x[1].-mb.x0s
  @. @views mb.em[5:8] = x[2].-mb.y0s
  return sum(@. @views mb.A.*exp.(mb.a.*mb.em[1:4].^2 + mb.b.*mb.em[1:4].*mb.em[5:8] + mb.c.*mb.em[5:8].^2))
end

function potential(system::AbstractSystem, mb::MullerBrown)
  @. @views mb.em[1:4] = system.x[1].-mb.x0s
  @. @views mb.em[5:8] = system.x[2].-mb.y0s
  return sum(@. @views mb.A.*exp.(mb.a.*mb.em[1:4].^2 + mb.b.*mb.em[1:4].*mb.em[5:8] + mb.c.*mb.em[5:8].^2))
end

"""
Force functions with return value
"""
function force(x::Vector{Float64}, mb::MullerBrown)
  @. @views em[1:4]  = x[1].-mb.x0s
  @. @views em[5:8]  = x[2].-mb.y0s
  @. @views em[9:12] = exp.(mb.a.*em[1:4].^2 + mb.b.*em[1:4].*em[5:8] + mb.c.*em[5:8].^2)
  fout::Vector{Float64} = zeros(Float64,2)
  f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*em[1:4] + mb.b.*em[5:8]).*em[9:12])
  f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*em[5:8] + mb.b.*em[1:4]).*em[9:12])
  return fout
end

function force(system::AbstractSystem, mb::MullerBrown)
  @. @views mb.em[1:4]  = system.x[1].-mb.x0s
  @. @views mb.em[5:8]  = system.x[2].-mb.y0s
  @. @views mb.em[9:12] = exp.(mb.a.*mb.em[1:4].^2 + mb.b.*mb.em[1:4].*mb.em[5:8] + mb.c.*mb.em[5:8].^2)
  fout::Vector{Float64} = zeros(Float64,2)
  f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*mb.em[1:4] + mb.b.*mb.em[5:8]).*mb.em[9:12])
  f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*mb.em[5:8] + mb.b.*mb.em[1:4]).*mb.em[9:12])
  return fout
end

"""
Force functions with in-place changing
"""
function force!(x::Vector{Float64}, f::Vector{Float64}, mb::MullerBrown)
  @. @views mb.em[1:4]  = x[1].-mb.x0s
  @. @views mb.em[5:8]  = x[2].-mb.y0s
  @. @views mb.em[9:12] = exp.(mb.a.*(mb.em[1:4].^2) + mb.b.*(mb.em[1:4].*mb.em[5:8]) + mb.c.*(mb.em[5:8].^2))
  f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*mb.em[1:4] + mb.b.*mb.em[5:8]).*mb.em[9:12])
  f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*mb.em[5:8] + mb.b.*mb.em[1:4]).*mb.em[9:12])
end

function force!(system::AbstractSystem, mb::MullerBrown)
  #@. @views mb.em[1:4]  = system.x[1].-mb.x0s
  #@. @views mb.em[5:8]  = system.x[2].-mb.y0s
  #@. @views mb.em[9:12] = exp.(mb.a.*(mb.em[1:4].^2) + mb.b.*(mb.em[1:4].*mb.em[5:8]) + mb.c.*(mb.em[5:8].^2))
  #mb.f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*mb.em[1:4] + mb.b.*mb.em[5:8]).*mb.em[9:12])
  #mb.f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*mb.em[5:8] + mb.b.*mb.em[1:4]).*mb.em[9:12])
  #@. @views mb.em[1:4]  = 0.63 .- mb.x0s
  #@. @views mb.em[5:8]  = 0.03 .- mb.y0s
  #@. @views mb.em[9:12] = exp.(mb.a.*(mb.em[1:4].^2) + mb.b.*(mb.em[1:4].*mb.em[5:8]) + mb.c.*(mb.em[5:8].^2))
  #mb.f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*mb.em[1:4] + mb.b.*mb.em[5:8]).*mb.em[9:12])
  #mb.f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*mb.em[5:8] + mb.b.*mb.em[1:4]).*mb.em[9:12])
  #println(mb.f)
  mb.em[1:4]  .= 0.63 .- mb.x0s
  mb.em[5:8]  .= 0.03 .- mb.y0s
  mb.em[9:12] .= exp.(mb.a.*(mb.em[1:4].^2) .+ mb.b.*(mb.em[1:4].*mb.em[5:8]) .+ mb.c.*(mb.em[5:8].^2))
  mb.f[1] = -sum( mb.A.*(2 .*mb.a.*mb.em[1:4] .+ mb.b.*mb.em[5:8]).*mb.em[9:12])
  mb.f[2] = -sum( mb.A.*(2 .*mb.c.*mb.em[5:8] .+ mb.b.*mb.em[1:4]).*mb.em[9:12])
  println(mb.em[1:4])
  println(mb.em[5:8])
  #println(mb.em[9:12])
  println( mb.A.*(2 .*mb.a.*mb.em[1:4]).*mb.em[9:12] )
  println( mb.A.*(mb.b.*mb.em[5:8]).*mb.em[9:12] )
  println( mb.A.*(2 .*mb.c.*mb.em[5:8]).*mb.em[9:12] )
  println( mb.A.*(mb.b.*mb.em[1:4]).*mb.em[9:12] )
  exit()
end
