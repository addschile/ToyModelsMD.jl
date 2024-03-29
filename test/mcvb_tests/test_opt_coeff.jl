include("../../src/ToyModelsMD.jl")
using .ToyModelsMD
using DelimitedFiles
using Plots

function Afunc(system::AbstractSystem,mm::MixedModel,ind::Int64,dt::Float64)
  lam::Float64 = 2.0e3
  pot::Float64 = potential(system,mm.potentials[1])
  ind==1500 && system.x[2]>0.7 && pot<-145.0 ? lam : 0.0
end

function cbfunc(xstore::Array,system::AbstractSystem,mm::MixedModel,ind::Int64,dt::Float64)
  @. @views xstore[ind,:] .= copy(system.x)
end

### define system ###
dt = 0.0001
steps = 1500
tf = steps*dt

# mueller-brown potential
mb = MullerBrown()

# gaussian force model
Mx = 21
My = 21
Mt = 21
shape = (Mx,My,Mt,2)
# Define gaussian basis centers and variance
mux = zeros(Float64,Mx)
muy = zeros(Float64,My)
mut = zeros(Float64,Mt)
for j=1:Mx
  mux[j] = -1.5+(j-1)*(1.5-(-1.5))/(Mx-1)
end
for j=1:My
  muy[j] = -0.5+(j-1)*(2-(-0.5))/(My-1)
end
for j in 1:Mt
  mut[j] = 1+(j*steps/(Mt-1)) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  mut[j] = (j*steps/(Mt-1)) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  mut[j] *= dt
end
sigx = zeros(Float64,Mx)
sigy = zeros(Float64,My)
sigt = zeros(Float64,Mt)
sigx[:] .= ((1.5-(-1.5))/(Mx-1)/2.0)^2 #variance
sigy[:] .= ((2-(-0.5))/(My-1)/2.0)^2 #variance
sigt[:] .= ((steps*1.0/(Mt-1))/2.0)^2 #variance
sigt[:] .*= dt^2
# read in optimal coefficients
x = readdlm("coeff0x.txt")
y = readdlm("coeff0y.txt")
cs = zeros(Float64,shape)
for i in 1:21
  cs[:,:,i,1] .= reshape(x[:,i],(21,21))'
  cs[:,:,i,2] .= reshape(y[:,i],(21,21))'
end
cs = reshape(cs, (21*21*21,2))
# define model - yay!
gm = GaussianModel2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,cs)

# mixed model
model = MixedModel(2,[mb,gm])

# thermostat and system
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(2,0.0,[0.63,0.03],zeros(Float64,2),langevin)

# define value baseline function
shape = (Mx,My,Mt)
vmux = zeros(Float64,Mx)
vmuy = zeros(Float64,My)
vmut = zeros(Float64,Mt)
for j=1:Mx
  vmux[j] = -1.5+(j-1)*(1.5-(-1.5))/(Mx-1)
end
for j=1:My
  vmuy[j] = -0.5+(j-1)*(2-(-0.5))/(My-1)
end
for j in 1:Mt
  vmut[j] = 1+(j-1)*steps*1.0/(Mt-1) # 1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
  vmut[j] *= dt
end
vsigx = zeros(Float64,Mx)
vsigy = zeros(Float64,My)
vsigt = zeros(Float64,Mt)
vsigx[:] .= ((1.5-(-1.5))/(Mx-1)/2.0)^2 #variance
vsigy[:] .= ((2-(-0.5))/(My-1)/2.0)^2 #variance
vsigt[:] .= ((steps*1.0/(Mt-1))/2.0)^2 #variance
vsigt[:] .*= dt^2
# read in optimal coefficients
v = readdlm("coeff0V.txt")
vcs = zeros(Float64,shape)
for i in 1:21
  vcs[:,:,i] .= reshape(v[:,i],(21,21))'
end
vcs = reshape(vcs, (21*21*21))
vbl = GaussianValueBaseline2D(Mx,My,Mt,mux,muy,mut,sigx,sigy,sigt,vcs)

# define callback function
mcvb = MCVBCallback(Afunc,gm,vbl)
# set up integrator and run!
integrator = StochasticEuler(system,model)
runtraj!(steps,dt,integrator,mcvb)
println(mcvb.dkl)
writedlm("gradf.txt",mcvb.gradf)
writedlm("gradv.txt",mcvb.gradv)
#mcvb.gradf .*= dt
#mcvb.gradv .*= dt

av_dx = readdlm("avishek_data/new_delomegax.txt")
av_dy = readdlm("avishek_data/new_delomegay.txt")
av_dv = readdlm("avishek_data/new_delV.txt")

println(sum(mcvb.gradf[1:Mx*My*Mt] .- av_dx))
println(sum(mcvb.gradf[Mx*My*Mt+1:end] .- av_dy))
println(sum(mcvb.gradv .- av_dv))

writedlm("diff_dx.txt",mcvb.gradf[1:Mx*My*Mt] .- av_dx)
writedlm("diff_dy.txt",mcvb.gradf[Mx*My*Mt+1:end] .- av_dy)
writedlm("diff_dv.txt",mcvb.gradv .- av_dv)
