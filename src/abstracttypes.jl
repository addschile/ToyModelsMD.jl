# models/potentials
abstract type AbstractPotential end
abstract type AbstractMixedModel <: AbstractPotential end

# systems
abstract type AbstractSystem end

# thermostats
abstract type AbstractThermostat end

# integrators
abstract type AbstractIntegrator end

# callbacks
abstract type AbstractCallback end

# reinforcement learning types
abstract type AbstractValueBaseline end
