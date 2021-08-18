### struct for defining a Gaussian force model
#mutable struct GaussianModel2D <: AbstractPotential
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
  cs::Matrix{Float64}
  f::Vector{Float64}
  em::Vector{Float64}
  function GaussianModel2D(nx::Int64,ny::Int64,nt::Int64,
                           mux::Vector{Float64},muy::Vector{Float64},mut::Vector{Float64},
                           sigx::Vector{Float64},sigy::Vector{Float64},sigt::Vector{Float64},
                           cs::Matrix{Float64})
    new(2,nx,ny,nt,mux,muy,mut,sigx,sigy,sigt,cs,zeros(Float64,2),zeros(Float64,(nx*ny*nt)))
  end
end

function getdimensionality(gm::GaussianModel2D)
  return 2
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
  fout[1] = sum( @views gm.cs[:,1] .* gm.em )
  fout[2] = sum( @views gm.cs[:,2] .* gm.em )
  return fout
end

"""
Force function with in-place changing
"""
function force!(sys::AbstractSystem, gm::GaussianModel2D)
  gm.em .= vec(kron(exp.(-0.5 .* (sys.t .- gm.mut).^2 ./ gm.sigt),
               kron(exp.(-0.5 .* (sys.x[2] .- gm.muy).^2 ./ gm.sigy),
                    exp.(-0.5 .* (sys.x[1] .- gm.mux).^2 ./ gm.sigx))))
  gm.f[1] = sum( @views gm.cs[:,1] .* gm.em )
  gm.f[2] = sum( @views gm.cs[:,2] .* gm.em )
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
  grad[:,1] .= gm.em
  grad[:,2] .= gm.em
end
