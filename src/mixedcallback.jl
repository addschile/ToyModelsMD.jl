"""
Mixed callback struct for handling multiple callback functions in a single run
"""
mutable struct MixedCallback <: AbstractCallback
    cbs::Vector{T} where T <: AbstractCallback
    function MixedCallback(cbs::Vector{T}) where T <: AbstractCallback
        new(cbs)
    end
end

function callback(cbs::MixedCallback,system::AbstractSystem,model::AbstractPotential,args...)
    for cb in cbs
        callback(cb,system,model,args)
    end
end