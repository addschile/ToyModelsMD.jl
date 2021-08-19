### ToyModelsMD.jl

A small Julia package to simulate molecular dynamics of toy models such as a single particle in the M\"{u}ller-Brown potential.

## Installation

As always the installation of Julia packages is quite simple, just open a REPL and follow these instructions
```
julia> ]
pkg> add https://github.com/addschile/ToyModelsMD.jl.git
```
and you should be all set. For MPI capabilities it is straightforward on a personal computer, the package manager should set you up just fine, but if you're using this package in an HPC environment, it is always recommended you follow the specific instructions for configuring an MPI build [here](https://juliaparallel.github.io/MPI.jl/stable/configuration/).

## Examples

Coming soon!

## Extensions

I've done my best to make this package usable as a playground for developing new algorithms in molecular dynamics for enhanced sampling of configurations or of trajectories. If you (yes you) have any ideas you'd like to contribute, go for it and let me know how it works out!