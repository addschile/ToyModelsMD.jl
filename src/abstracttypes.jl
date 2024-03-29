# models/potentials
abstract type AbstractPotential end
abstract type AbstractMixedModel <: AbstractPotential end
abstract type AbstractSinglePotential <: AbstractPotential end
abstract type AbstractTrainablePotential <: AbstractSinglePotential end

# systems
abstract type AbstractSystem end
abstract type AbstractThermostattedSystem <: AbstractSystem end
abstract type AbstractActiveSystem <: AbstractThermostattedSystem end

# thermostats
abstract type AbstractThermostat end
abstract type AbstractLangevin <: AbstractThermostat end

# integrators
abstract type AbstractIntegrator end

# callbacks
abstract type AbstractCallback end

# reinforcement learning types
abstract type AbstractValueBaseline end

# optimizer types
abstract type AbstractOptimizer end
