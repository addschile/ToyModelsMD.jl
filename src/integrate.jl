function runtraj!(nsteps::Int64,dt::Float64,system::System)
end

function integrate!(nsteps::Int64,dt::Float64,system::System)
  # forward euler integrator
  vector<double> f = force(x);
  for (int i=0; i<2; i++){
    x[i] += dt*f[i] + nd(gen);
  }
  return;
end

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
