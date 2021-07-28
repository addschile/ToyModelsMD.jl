abstract type AbstractPotential end

struct DummyPotential <: AbstractPotential; end

### struct for defining the Muller-Brown potential
struct MullerBrown <: AbstractPotential
  A::Array{Float64}
  a::Array{Float64}
  b::Array{Float64}
  c::Array{Float64}
  x0s::Array{Float64}
  y0s::Array{Float64}
  function MullerBrown()
    new([-200.,-100.,-170.,-15.],[-1.,-1.,-6.5,0.7],[0.,0.,11.,0.6],[-10.,-10.,-6.5,0.7],[1.,0.,-0.5,-1.],[0.,0.5,1.5,1.])
  end
end

function genextramem(mb::MullerBrown)
  return zeros(Float64,12)
end

function getdimensionality(mb::MullerBrown)
  return 2
end

function potential(x::Array{Float64}, mb::MullerBrown)
  dx::Array{Float64} = x[1].-mb.x0s
  dy::Array{Float64} = x[2].-mb.y0s
  return sum(mb.A.*exp.(mb.a.*dx.^2 + mb.b.*dx.*dy + mb.c.*dy.^2))
end

function potential(x::Array{Float64}, em::Array{Float64}, mb::MullerBrown)
  @. @views em[1:4] = x[1].-mb.x0s
  @. @views em[5:8] = x[2].-mb.y0s
  return sum(@. @views mb.A.*exp.(mb.a.*em[1:4].^2 + mb.b.*em[1:4].*em[5:8] + mb.c.*em[5:8].^2))
end

function force(x::Array{Float64}, mb::MullerBrown)
  dx::Array{Float64}    = x[1].-mb.x0s
  dy::Array{Float64}    = x[2].-mb.y0s
  epart::Array{Float64} = exp.(mb.a.*dx.^2 + mb.b.*dx.*dy + mb.c.*dy.^2)
  fout::Array{Float64}  = zeros(Float64,2)
  fout[1] -= sum(mb.A.*(2 .*mb.a.*dx + mb.b.*dy).*epart)
  fout[2] -= sum(mb.A.*(2 .*mb.c.*dy + mb.b.*dx).*epart)
end

function force!(x::Array{Float64}, em::Array{Float64}, mb::MullerBrown)
  @. @views em[1:4]  = x[1].-mb.x0s
  @. @views em[5:8]  = x[2].-mb.y0s
  @. @views em[9:12] = exp.(mb.a.*em[1:4].^2 + mb.b.*em[1:4].*em[5:8] + mb.c.*em[5:8].^2)
  fout::Array{Float64} = zeros(Float64,2)
  f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*em[1:4] + mb.b.*em[5:8]).*em[9:12])
  f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*em[5:8] + mb.b.*em[1:4]).*em[9:12])
  return fout
end

function force!(x::Array{Float64}, f::Array{Float64}, em::Array{Float64}, mb::MullerBrown)
  @. @views em[1:4]  = x[1].-mb.x0s
  @. @views em[5:8]  = x[2].-mb.y0s
  @. @views em[9:12] = exp.(mb.a.*em[1:4].^2 + mb.b.*em[1:4].*em[5:8] + mb.c.*em[5:8].^2)
  f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*em[1:4] + mb.b.*em[5:8]).*em[9:12])
  f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*em[5:8] + mb.b.*em[1:4]).*em[9:12])
end
