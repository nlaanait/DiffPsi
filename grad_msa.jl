using CuArrays, CUDAdrv, CUDAnative
using LinearAlgebra, Flux
using Flux: Optimise
using Zygote
using FFTW

# FFTW.set_num_threads(40)

# MSA imports
include("MSA.jl")
using .MSA

CUDAnative.device!(3)
CuArrays.allowscalar(true)


mutable struct ForwardModel
    psi::AbstractArray
    potential::AbstractArray
    name::String
end

function forward(model::ForwardModel)
    psi_buff = Zygote.Buffer(model.potential)
    psi_buff[:,:,:] = model.psi[:,:,:]
    psi_buff = MSA.multislice(model.psi, psi_buff, model.potential, 1, 1, 1)
    model.psi = copy(psi_buff)
    psi_out = model.psi[:,:,end]
    return abs2.(psi_out)
end

function loss(model::ForwardModel, psi2_trg)
    res = norm(forward(model) .- psi2_trg, 2)
    return res
end

function sgd_update!(model::ForwardModel, grads, η = 0.01; verbose=false)
    if verbose 
        @info("Gradients:
            ∂V=$(round.(grads.potential; digits=2))
            ∂ѱ=$(round.(grads.psi; digits=2))")
        @info("before sgd update:
        V'=$(round.(model.potential; digits=2))
        ѱ'=$(round.(model.psi; digits=2))")
    end
    model.psi .-= η .* grads.psi
    model.potential .-= η .* grads.potential
    if verbose
        @info("after sgd update:
        V'=$(round.(model.potential; digits=2))
        ѱ'=$(round.(model.psi; digits=2))")
    end
end


k_size = (64,64)
num_slices= 32 
@info("Initiate Wavefunction and Scattering Potential...") 
psi_trg = randn(ComplexF32, k_size) 
cpy_trg = copy(psi_trg)
V_trg = im * randn(Float32, (k_size..., num_slices)) 
MSA.multislice!(psi_trg, V_trg, 1, 1, 1)
psi2_trg = abs2.(psi_trg)
@info("Target Values:")
# println("V=$(round.(V_trg; digits=2))
# ѱ=$(round.(cpy_trg; digits=2))
# psi_out = $(round.(psi2_trg; digits=2))") 

@info("Initiate Forward Model...")
scale = Float32(5e-1)
psi_in = randn(ComplexF32, size(V_trg))
V_in = im * randn(Float32, size(V_trg))
model = ForwardModel(psi_in, V_in, "Slices:1")
@info("Initial Model Values:")
# print("V'=$(round.(model.potential; digits=2))
# ѱ'=$(round.(model.psi; digits=2))
# psi_out = $(round.(forward(model); digits=2))")

# model gradients
@info("Differentiating Forward Model...")
grads = gradient(model) do m
    return loss(m, psi2_trg)
end
grads = grads[1][]
@info("Gradients:")
# println("
#     ∂V=$(round.(grads.potential; digits=2))
#     ∂ѱ=$(round.(grads.psi; digits=2))")

opt = ADAM(1e-3)
@info("Running train loop")
idx = 0
loss_val = loss(model, psi2_trg)
max_iter = 5e3
num_logs = 10
verbose = false
@time while idx < max_iter && loss_val > 1e-4
    if mod(idx, max_iter ÷ num_logs) == 0
        Zygote.@nograd loss_val = loss(model, psi2_trg)
        @info("Iteration=$(idx), Loss=$loss_val")
    end
    grads= Zygote.gradient(model) do m
        return loss(m, psi2_trg)
    end
    grads = grads[1][]
#     sgd_update!(model, grads, 1e-1; verbose=false)
    Optimise.update!(opt, model.psi, grads.psi)
    Optimise.update!(opt, model.potential, grads.potential)
    global idx += 1
end