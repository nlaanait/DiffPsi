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
    amplitude::AbstractArray
    phase::AbstractArray
    potential::AbstractArray
    name::String
end

function forward(model::ForwardModel)
    psi = model.amplitude * exp.(model.phase)
    psi_buff = Zygote.Buffer(psi)
    psi_buff[:,:] = psi
    psi_buff = MSA.multislice(psi_buff, model.potential, 0.1, 0.1, 10.0)
    # psi_buff = MSA.multislice(psi, psi_buff, model.potential, 0.1, 0.1, 10.0)
    return abs2.(copy(psi_buff))
end

function loss(model::ForwardModel, psi2_trg)
    res = norm(forward(model) .- psi2_trg, 2)
    return res
end

# Initiate V_trg and psi_trg
k_size = (2,2)
num_slices= 1
@info("Initiate Wavefunction and Scattering Potential...") 
amp_trg = randn(Float32, k_size)
phase_trg = im * randn(Float32, k_size)
psi_trg = amp_trg * exp.(phase_trg)
V_trg = im * randn(Float32, (k_size..., num_slices)) 
MSA.multislice!(psi_trg, V_trg, 0.1, 0.1, 10.0)
psi2_trg = abs2.(psi_trg) 

# build forward model
@info("Initiate Forward Model...")
scale = Float32(1e-32)
amp_in = amp_trg + scale * randn(Float32, size(psi_trg))
phase_in = phase_trg + scale * im * randn(Float32, size(psi_trg))
V_in = V_trg + scale * im * randn(Float32, size(V_trg))
model = ForwardModel(amp_in, phase_in, V_in, "Slices:1")

# model gradients
@info("Differentiating Forward Model...")
grads = gradient(model) do m
    return loss(m, psi2_trg)
end
grads = grads[1][]

@info("Gradients:
    𝛻V=$(round.(grads.potential; digits=2))
    𝛻A=$(round.(grads.amplitude; digits=2))
    𝛻ϕ=$(round.(grads.phase; digits=2))")

function sgd_update!(model::ForwardModel, grads, η = 0.01)
    model.amplitude .-= η .* real.(grads.amplitude)
    model.phase     .-= η .* im .* real.(grads.phase)
    model.potential .-= η .* im .* real.(grads.potential)
end

opt = Flux.ADAM(0.001)
# opt = ADADelta()
@info("Running train loop")
idx = 0
loss_val = loss(model, psi2_trg)
max_iter = 1e4
num_logs = 5
verbose = false
@time while idx < max_iter
    if mod(idx, max_iter ÷ num_logs) == 0
        loss_val = loss(model, psi2_trg)
        @info("Iteration=$(idx), Loss=$loss_val")
    #     @info("Learned parameters:\nV=$(round.(model.potential; digits=2))\n 
    #     A=$(round.(model.amplitude; digits=2))\n
    #     ϕ=$(round.(model.phase; digits=2))\n")

    #     @info("True parameters:\nV=$(round.(V_trg; digits=2))\n 
    #     A=$(round.(amp_trg; digits=2))\n
    #     ϕ=$(round.(phase_trg; digits=2))")
    end
    grads= Zygote.gradient(model) do m
        return loss(m, psi2_trg)
    end
    grads = grads[1][]
    if verbose
        @info("Gradients:
            𝛻V=$(round.(grads.potential; digits=2))
            𝛻A=$(round.(grads.amplitude; digits=2))
            𝛻ϕ=$(round.(grads.phase; digits=2))")
    end
    sgd_update!(model, grads)
    # Optimise.update!(opt, model.amplitude, real.(grads.amplitude))
    # Optimise.update!(opt, model.phase, im * real.(grads.phase))
    # Optimise.update!(opt, model.potential, im * real.(grads.potential))
    global idx += 1
end
@info("Learned parameters:
V=$(round.(model.potential; digits=2))
V_trg=$(round.(V_trg; digits=2))
A=$(round.(model.amplitude; digits=2))
A_trg=$(round.(amp_trg; digits=2))
ϕ=$(round.(model.phase; digits=2))
ϕ_trg=$(round.(phase_trg; digits=2))")
