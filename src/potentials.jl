struct DummyPotential <: AbstractPotential; end

function addforce!(dt::Float64,system::AbstractSystem,model::AbstractPotential)
  system.x .+= dt .* model.f
end
