include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using BenchmarkTools
using Profile

#function force!(sys::AbstractSystem, gm::GaussianModel2D)
#  val::Float64 = 0.0
#  for i in 1:gm.nx
#    for j in 1:gm.ny
#      for k in 1:gm.nt
#        val = exp(-0.5*(sys.x[1]-gm.mux[i])^2/gm.sigx[i])
#        gm.em[i,j,k] = val
#        val = exp(-0.5*(sys.x[2]-gm.muy[j])^2/gm.sigy[j])
#        gm.em[i,j,k] *= val
#        val = exp(-0.5*(sys.t-gm.mut[k])^2/gm.sigt[k])
#        gm.em[i,j,k] *= val
#        #gm.em[i,j,k]  = exp(-0.5*(sys.x[1]-gm.mux[i])^2/gm.sigx[i])
#        #gm.em[i,j,k] *= exp(-0.5*(sys.x[2]-gm.muy[j])^2/gm.sigy[j])
#        #gm.em[i,j,k] *= exp(-0.5*(sys.t-gm.mut[k])^2/gm.sigt[k])
#      end
#    end
#  end
#  gm.f[1] += sum( @. @views gm.cs[:,:,:,1]*gm.em[:,:,:] )
#  gm.f[2] += sum( @. @views gm.cs[:,:,:,2]*gm.em[:,:,:] )
#end

function force!(x::Array{Float64},t::Float64,f::Array{Float64},nx::Int64,ny::Int64,nt::Int64,
                mux::Array{Float64},muy::Array{Float64},mut::Array{Float64},
                sigx::Array{Float64},sigy::Array{Float64},sigt::Array{Float64},cs::Array{Float64},em::Array{Float64})
  for i in 1:nx
    @. @views em[i,:,:] .= exp(-0.5*(x[1]-mux[i])^2/sigx[i])
    #@views em[i,:,:] .= exp(-0.5*(x[1]-mux[i])^2/sigx[i])
  end
  for i in 1:ny
    @. @views em[:,i,:] .*= exp(-0.5*(x[2]-muy[i])^2/sigy[i])
    #@views em[:,i,:] .*= exp(-0.5*(x[2]-muy[i])^2/sigy[i])
  end
  for i in 1:nt
    #@. @views em[:,:,i] .*= exp(-0.5*(t-mut[i])^2/sigt[i])
    @views em[:,:,i] .*= exp(-0.5*(t-mut[i])^2/sigt[i])
  end
  #f[1] = sum( @. @views cs[:,:,:,1]*em[:,:,:] )
  #f[2] = sum( @. @views cs[:,:,:,2]*em[:,:,:] )
  f[1] = sum( @views cs[:,:,:,1]*em[:,:,:] )
  f[2] = sum( @views cs[:,:,:,2]*em[:,:,:] )
end

function forcewviews!(sys::AbstractSystem, gm::GaussianModel2D)
  for i in 1:gm.nx
    @. @views gm.em[i,:,:] .= exp(-0.5*(sys.x[1]-gm.mux[i])^2/gm.sigx[i])
  end
  for i in 1:gm.ny
    @. @views gm.em[:,i,:] .*= exp(-0.5*(sys.x[2]-gm.muy[i])^2/gm.sigy[i])
  end
  for i in 1:gm.nt
    @. @views gm.em[:,:,i] .*= exp(-0.5*(sys.t-gm.mut[i])^2/gm.sigt[i])
  end
  gm.f[1] = sum( @. @views gm.cs[:,:,:,1]*gm.em[:,:,:] )
  gm.f[2] = sum( @. @views gm.cs[:,:,:,2]*gm.em[:,:,:] )
end

function force!(sys::AbstractSystem, gm::GaussianModel2D)
  for i in 1:gm.nx
    gm.em[i,:,:] .= exp(-0.5*(sys.x[1]-gm.mux[i])^2/gm.sigx[i])
  end
  for i in 1:gm.ny
    gm.em[:,i,:] .*= exp(-0.5*(sys.x[2]-gm.muy[i])^2/gm.sigy[i])
  end
  for i in 1:gm.nt
    gm.em[:,:,i] .*= exp(-0.5*(sys.t-gm.mut[i])^2/gm.sigt[i])
  end
  gm.f[1] = sum( @. @views gm.cs[:,:,:,1]*gm.em[:,:,:] )
  gm.f[2] = sum( @. @views gm.cs[:,:,:,2]*gm.em[:,:,:] )
end

### define system ###
# mueller-brown potential
mb = MullerBrown()
# gaussian force model
Mx = 21
My = 21
Mt = 21
shape = (Mx,My,Mt,2)
mux = rand(Float64,21)
muy = rand(Float64,21)
mut = rand(Float64,21)
sigx = rand(Float64,21)
sigy = rand(Float64,21)
sigt = rand(Float64,21)
cs = zeros(Float64,shape)
gm = GaussianModel2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,cs)
# mixed model
model = MixedModel(2,[mb,gm])
# thermostat and system
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

#@btime @. @views gm.em[1,:,:] .= exp(-0.5*(system.x[1]-gm.mux[1])^2/gm.sigx[1])
#@btime @. @views gm.em[:,1,:] .*= exp(-0.5*(system.x[2]-gm.muy[1])^2/gm.sigy[1])
#@btime @. @views gm.em[:,:,1] .*= exp(-0.5*(system.t-gm.mut[1])^2/gm.sigt[1])
#@btime gm.em[1,:,:] .= exp(-0.5*(system.x[1]-gm.mux[1])^2/gm.sigx[1])
#@btime gm.em[:,1,:] .*= exp(-0.5*(system.x[2]-gm.muy[1])^2/gm.sigy[1])
#@btime gm.em[:,:,1] .*= exp(-0.5*(system.t-gm.mut[1])^2/gm.sigt[1])

x = [0.63,0.03]
@btime exp(-0.5*(x[1]-mux[1])^2/sigx[1])
@btime exp(-0.5*(system.x[1]-gm.mux[1])^2/gm.sigx[1])
val = exp(-0.5*(x[1]-mux[1])^2/sigx[1])
@btime @. @views gm.em[1,:,:] .= val
@btime gm.em[1,:,:] .= val
@btime @. @views gm.em[1,:,:] .= exp(-0.5*(system.x[1]-gm.mux[1])^2/gm.sigx[1])
@btime gm.em[1,:,:] .= exp(-0.5*(system.x[1]-gm.mux[1])^2/gm.sigx[1])
#Profile.print()
#@profile gm.em[1,:,:] .= exp(-0.5*(system.x[1]-gm.mux[1])^2/gm.sigx[1])
#Profile.print()

#@btime forcewviews!(system,gm)
#@btime force!(system,gm)
#@btime force!(system.x,system.t,gm.f,gm.nx,gm.ny,gm.nt,
#              gm.mux,gm.muy,gm.mut,
#              gm.sigx,gm.sigy,gm.sigt,gm.cs,gm.em)

#println("first operation")
#x = [0.63,0.03]
#em = zeros(Float64,(Mx,My,Mt))
#@btime for i in 1:Mx
#  #@. @views em[i,:,:] .= exp(-0.5*(x[1]-mux[i])^2/sigx[i])
#  em[i,:,:] .= exp(-0.5*(x[1]-mux[i])^2/sigx[i])
#end
#
#println("second operation")
#@btime for i in 1:gm.ny
#  @. @views gm.em[:,i,:] .*= exp(-0.5*(system.x[2]-gm.muy[i])^2/gm.sigy[i])
#end
#
#println("third operation")
#@btime for i in 1:gm.nt
#  @. @views gm.em[:,:,i] .*= exp(-0.5*(system.t-gm.mut[i])^2/gm.sigt[i])
#end
#
#println("fourth operation")
#@btime gm.f[1] = sum( @. @views gm.cs[:,:,:,1]*gm.em[:,:,:] )
#
#println("fifth operation")
#@btime gm.f[2] = sum( @. @views gm.cs[:,:,:,2]*gm.em[:,:,:] )
