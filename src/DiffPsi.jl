module DiffPsi

using Zygote, Flux, FFTW, LinearAlgebra
using Flux: Optimise

include("msa.jl")
include("grad_msa.jl")
include("utils.jl")


end
