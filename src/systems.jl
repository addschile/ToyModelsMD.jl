struct DummySystem <: AbstractSystem; end

"""
Generic struct for a system
"""
mutable struct System <: AbstractSystem
  dim::Int64
  t::Float64
  x::Vector{Float64}
  f::Vector{Float64}
  function System(dim::Int64,t::Float64,x::Vector{Float64},f::Vector{Float64})
    new(dim,t,x,f)
  end
end

function System(model::AbstractPotential)
  dim::Int64 = getdimensionality(model)
  x::Vector{Float64} = zeros(dim)
  f::Vector{Float64} = zeros(dim)
  System(dim,0.0,x,f)
end

"""
Generic struct for a system with a thermostat
"""
mutable struct ThermostattedSystem <: AbstractThermostattedSystem
  dim::Int64
  t::Float64
  x::Vector{Float64}
  f::Vector{Float64}
  thermostat::AbstractThermostat
  function ThermostattedSystem(dim::Int64,t::Float64,x::Vector{Float64},f::Vector{Float64},thermostat::AbstractThermostat)
    new(dim,t,x,f,thermostat)
  end
end

function ThermostattedSystem(model::AbstractPotential,thermostat::AbstractThermostat)
  dim::Int64 = getdimensionality(model)
  x::Vector{Float64} = zeros(dim)
  f::Vector{Float64} = zeros(dim)
  ThermostattedSystem(dim,0.0,x,f,thermostat)
end

"""
Generic struct for a system with a thermostat
"""
mutable struct ThermostattedSystem <: AbstractThermostattedSystem
  dim::Int64
  t::Float64
  x::Vector{Float64}
  f::Vector{Float64}
  thermostat::AbstractThermostat
  function ThermostattedSystem(dim::Int64,t::Float64,x::Vector{Float64},f::Vector{Float64},thermostat::AbstractThermostat)
    new(dim,t,x,f,thermostat)
  end
end

function ThermostattedSystem(model::AbstractPotential,thermostat::AbstractThermostat)
  dim::Int64 = getdimensionality(model)
  x::Vector{Float64} = zeros(dim)
  f::Vector{Float64} = zeros(dim)
  ThermostattedSystem(dim,0.0,x,f,thermostat)
end

#"""
#Struct for an overdamped system with a langevin thermostat
#"""
#mutable struct OverdampedLangevinSystem <: AbstractThermostattedSystem
#  dim::Int64
#  t::Float64
#  x::Vector{Float64}
#  f::Vector{Float64}
#  thermostat::AbstractLangevin
#  function OverdampedLangevinSystem(dim::Int64,t::Float64,x::Vector{Float64},f::Vector{Float64},thermostat::AbstractLangevin)
#    new(dim,t,x,f,thermostat)
#  end
#end
#
#function OverdampedLangevinSystem(model::AbstractPotential,thermostat::AbstractLangevin)
#  dim::Int64 = getdimensionality(model)
#  x::Vector{Float64} = zeros(dim)
#  f::Vector{Float64} = zeros(dim)
#  ThermostattedSystem(dim,0.0,x,f,thermostat)
#end

"""
Functions for interacting with the force
"""
function zeroforce!(system::AbstractSystem)
  system.f .= 0.0
end

function addforce!(dt::Float64,system::AbstractSystem,model::AbstractSinglePotential)
  system.x .+= dt .* model.f
end

# TODO maybe take this function out since it's a repeat of the one above
function addforce!(dt::Float64,system::ThermostattedSystem,model::AbstractSinglePotential)
  system.x .+= dt .* model.f ./ system.thermostat.gamma
#  system.x .+= dt .* model.f
end

# TODO do unit tests on Overdamped system before allowing this
#function addforce!(dt::Float64,system::OverdampedLangevinSystem,model::AbstractPotential)
#  system.x .+= dt .* model.f ./ system.thermostat.gamma
#end