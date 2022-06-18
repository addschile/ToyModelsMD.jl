include("../../src/ToyModelsMD.jl")
using .ToyModelsMD

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

@btime force!(system,gm)

