using Random

abstract type System end
abstract type Potential end
abstract type Thermostat end

### struct for defining the Muller-Brown potential
struct MullerBrown <: Potential
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
  return fout;
end


function force!(x::Array{Float64}, em::Array{Float64}, mb::MullerBrown)
  @. @views em[1:4]  = x[1].-mb.x0s
  @. @views em[5:8]  = x[2].-mb.y0s
  @. @views em[9:12] = exp.(mb.a.*em[1:4].^2 + mb.b.*em[1:4].*em[5:8] + mb.c.*em[5:8].^2)
  fout::Array{Float64} = zeros(Float64,2)
  f[1] = @. @views -sum(mb.A.*(2 .*mb.a.*em[1:4] + mb.b.*em[5:8]).*em[9:12])
  f[2] = @. @views -sum(mb.A.*(2 .*mb.c.*em[5:8] + mb.b.*em[1:4]).*em[9:12])
  return fout;
end

function force!(x::Array{Float64}, f::Array{Float64}, em::Array{Float64}, mb::MullerBrown)
  @. @views em[1:4]  = x[1].-mb.x0s
  @. @views em[5:8]  = x[2].-mb.y0s
  @. @views em[9:12] = exp.(mb.a.*em[1:4].^2 + mb.b.*em[1:4].*em[5:8] + mb.c.*em[5:8].^2)
  f[1] = -sum(@. @views mb.A.*(2 .*mb.a.*em[1:4] + mb.b.*em[5:8]).*em[9:12])
  f[2] = -sum(@. @views mb.A.*(2 .*mb.c.*em[5:8] + mb.b.*em[1:4]).*em[9:12])
end

### struct for dealing with Langevin dynamics
mutable struct Langevin <: Thermostat
  T::Float64
  gamma::Float64
  rng::AbstractRNG
  function Langevin(T::Float64,gamma::Float64,seed::UInt64)
    new(T,gamma,MersenneTwister(seed))
  end


### struct for handling the system and its components
mutable struct ThermostattedSystem <: System
  model::Potential
  thermostat::Thermostat
  x::Array{Float64}
  f::Array{Float64}
  em::AbstractArray
  function ThermostattedSystem(model::Potential,thermostat::Thermostat)
    dim::Int64 = getdimensionality(model)
    x::Array{Float64} = zeros(dim)
    f::Array{Float64} = zeros(dim)
    em::AbstractArray = genextramem(model)
  end
end

#function integrate!(nsteps::Int64,dt::Float64,system::System)
#  # forward euler integrator
#  vector<double> f = force(x);
#  for (int i=0; i<2; i++){
#    x[i] += dt*f[i] + nd(gen);
#  }
#  return;
#end

mb = MullerBrown()
langevin = Langevin(1.0,1.0)
system = ThermostattedSystem(mb,langevin)
#x = [0.,0.]
#em = genextramem(mb)
#f = [0.,0.]

#  return 
#  double pot = 0.;
#  for (int i=0; i<4; i++){
#    double dx = x[0]-x0s[i];
#    double dy = x[1]-y0s[i];
#    pot += A[i]*exp(a[i]*dx*dx + b[i]*dx*dy + c[i]*dy*dy);
#  }
#  return pot;
#end
#
#
#void integrate(double dt,vector<double> &x,
#               default_random_engine &gen,
#               normal_distribution<double> &nd){
#  /* forward euler integrator */
#  vector<double> f = force(x);
#  for (int i=0; i<2; i++){
#    x[i] += dt*f[i] + nd(gen);
#  }
#  return;
#}
#
#void integrate_traj(int tobs,double dt, vector<double> &x,
#                    vector<vector<double> > &traj,
#                    default_random_engine &gen,
#                    normal_distribution<double> &nd){
#  for (int i=0; i<tobs; i++){
#    for (int j=0; j<2; j++){
#      traj[i][j] = x[j];
#    }
#    integrate(dt,x,gen,nd);
#  }
#  return;
#}
#
#int main(int argc, char* argv[]){
#
#  /* simulation parameters */
#  double dt = 0.0001;
#  double T = 15.;
#  int ntraj = 1;
#  int tobs = (int)(T/dt);
#
#  /* random number generator stuff */
#  default_random_engine gen(time(0));
#  normal_distribution<double> nd(0.,dt);
#
#  /* initialize trajectory and trajectory observable */
#  for (int i=0; i<ntraj; i++){
#    vector<double> x = {0.63,0.03};
#    vector<vector<double> > traj( tobs , vector<double> (2, 0));
#    integrate_traj(tobs,dt,x,traj,gen,nd);
#    for (int j=0; j<tobs; j++){
#      cout << traj[j][0] << " " << traj[j][1] << endl;
#    }
#  }
#
#  return 0;
#}
