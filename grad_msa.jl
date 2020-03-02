using CuArrays, CUDAdrv, CUDAnative
using LinearAlgebra
using Zygote
using FFTW

# MSA imports
include("MSA.jl")
using .MSA

CUDAnative.device!(3)
# CuArrays.allowscalar(false)

mutable struct ForwardModel
    amplitude::AbstractArray
    phase::AbstractArray
    potential::AbstractArray
    name::String
end

# function forward(model::ForwardModel)
#     num_slices = 10
#     psi_buff = Zygote.Buffer(model.psi_in)
#     psi_buff[:,:] = model.psi_in
#     k = randn(size(model.psi_in))
#     fresnel_op = MSA.build_fresnelPropagator(k)
#     for i in 1:size(model.potential,3)
#         trans = MSA.build_transmPropagator(model.potential[:,:,i])
#         psi_buff =  ifft(fresnel_op .* fft(copy(psi_buff) * trans)/sum(size(model.psi_in)))
#     end
#     return abs2.(copy(psi_buff))
# end

function forward(model::ForwardModel)
    psi = model.amplitude * exp.( 1.f0 * im * model.phase)
    psi_buff = Zygote.Buffer(psi)
    psi_buff[:,:] = psi
    psi_buff = MSA.multislice(psi_buff, model.potential, 0.1, 0.1, 10.0)
    return abs2.(copy(psi_buff))
end

function loss(model::ForwardModel, psi2_trg)
    res = norm(forward(model) .- psi2_trg, 2)
    println(res)
    return res
end

# Initiate V_trg and psi_trg
k_size = (4,4)
num_slices= 1
@info("Initiate Wavefunction and Scattering Potential...") 
# psi_trg = CuArrays.ones(ComplexF32, k_size) 
psi_trg = ones(Float32, k_size) .* im
# V_trg = CuArrays.ones(ComplexF32, (k_size..., num_slices)) 
V_trg = ones(Float32, (k_size..., num_slices)) 
MSA.multislice!(psi_trg, V_trg, 0.1, 0.1, 10.0)
psi2_trg = abs2.(psi_trg) 

# build forward model
@info("Initiate Forward Model...")
# psi_in = CuArrays.randn(Float32, k_size) .* ComplexF32( 1. * im)
# psi_in .+= CuArrays.randn(Float32, k_size)
# V_in = CuArrays.randn(Float32, (k_size..., num_slices))
psi_in = psi_trg + 1e1 * randn(ComplexF32, size(psi_trg))
amp_in = randn(Float32, size(psi_trg))
phase_in = randn(Float32, size(psi2_trg))
V_in = randn(Float32, size(V_trg))
# V_in = ones(ComplexF32, (k_size..., num_slices))
# V_in = randn(ComplexF32, (k_size..., num_slices))
model = ForwardModel(amp_in, phase_in, V_in, "Slices:1")

# model gradients
@info("Differentiating Forward Model...")
grads = gradient(model) do m
    return loss(m, psi2_trg)
end

grads = grads[1]
grads = grads[]
# println(size(grads.psi_in), size(grads.potential))
# @info grads

function sgd_update!(model::ForwardModel, grads, η = 0.001)
    model.amplitude .-= η .* abs2.(grads.amplitude)
    model.phase -= η * abs2.(grads.phase)
    model.potential -= η * abs2.(grads.potential)
end


@info("Running train loop")
for idx in 1:10000
    grads = Zygote.gradient(m -> loss(m, psi2_trg), model)[1][]
    sgd_update!(model, grads)
end

@info("Learned parameters:\n V=$(round.(model.potential; digits=2))\n 
                A=$(round.(model.amplitude; digits=2)) \n
                ϕ=$(round.(model.phase; digits=2))")