mutable struct Callback <: AbstractCallback
  every::Int64
  func::Function
  function Callback(every::Int64,func::Function)
    new(every,func)
  end
end
Callback(func::Function) = Callback(1,func)

function callback(cb::AbstractCallback,system::AbstractSystem,model::AbstractPotential,args...)
  cb.func(system,model,args)
end
