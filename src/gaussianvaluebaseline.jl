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
  cs::Vector{Float64}
  em::Vector{Float64}
  function GaussianValueBaseline2D(nx::Int64,ny::Int64,nt::Int64,
                                   mux::Vector{Float64},muy::Vector{Float64},mut::Vector{Float64},
                                   sigx::Vector{Float64},sigy::Vector{Float64},sigt::Vector{Float64},
                                   cs::Vector{Float64})
    new(nx,ny,nt,mux,muy,mut,sigx,sigy,sigt,cs,zeros(Float64,size(cs)))
  end
end

function gradshape(vbl::GaussianValueBaseline2D)
  return size(vbl.cs)
end

function callvbl(sys::AbstractSystem,vbl::GaussianValueBaseline2D)
  vbl.em .= vec(kron(exp.(-0.5 .* (sys.t .- vbl.mut).^2 ./ vbl.sigt),
                kron(exp.(-0.5 .* (sys.x[2] .- vbl.muy).^2 ./ vbl.sigy),
                     exp.(-0.5 .* (sys.x[1] .- vbl.mux).^2 ./ vbl.sigx))))
  return sum( vbl.cs.*vbl.em )
end

"""
Gradient wrt parameters for gaussian value baseline
"""
function gradient!(grad::Array{Float64},sys::AbstractSystem,vbl::GaussianValueBaseline2D)
  grad .= vbl.em
end
