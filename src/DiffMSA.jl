module DiffMSA

using Zygote, FFTW, Flux, LinearAlgebra

include("MSA.jl")
using .MSA


function FFT_op(x, plan_mat)
    return plan_mat * copy(x)
end

function IFFT_op(x, plan_mat)
    return plan_mat * copy(x)
end

Zygote.@adjoint FFT_op(x) = Zygote.pullback(fft, x)
Zygote.@adjoint IFFT_op(x) = Zygote.pullback(ifft, x)

function fft_shift(x)
    Zygote.nograd() do
        shifted = fftshift(x)
    end
    return shifted
end

mutable struct ForwardModel
    phase::AbstractArray
    potential::AbstractArray
    simParams::MSA.SimulationState
    name::String
end

function forward(model::ForwardModel, args)
    psi_k_amp, k_x, k_y, k_arr, pos = args
    psi_k = psi_k_amp .* exp.(-1.0f0 * im .* model.phase)
    probes = build_probes(psi_k, k_x, k_y, pos)
    psi_out = multislice(probes, model.potential, k_arr, model.simParams)
    psi2_out = abs.(psi_out)
    return psi2_out
end

function loss(model::ForwardModel, psi2_trg;  constrained=false)
    psi2_pred = forward(model, args)
    mse = norm(psi2_pred  .- psi2_trg, 2)
    if constrained:
        v_constr = norm(imag.(model.potential), 2) # force V to be real
        psi_constr = norm(imag.(model.phase), 2) # force phase to be real := psi modulus does not change
        return mse + v_constr + psi_constr
    end
    return mse 
end
    
function build_probes(psi_k, k_x, k_y, pos::MSA.Scan)
    psi_buff = Zygote.Buffer(psi_k)
    psi_k, k_x, k_y = map( x -> repeat(x, outer=(1, 1, length(pos.x))), (psi_k, k_x, k_y))
    kr = k_x .* x + k_y .* y
    phase_shift = exp.(2.f0 * pi * im .* kr)
    psi_buff[:,:,:] = ifft(psi_k .* phase_shift, [1,2])
    # looping appears to be necessary here- otherwise buffer becomes frozen
    for idx in 1:length(x)
        psi_buff[:,:,idx] = fftshift(copy(psi_buff[:,:,idx]))
    end
    norm = sqrt.(sum(abs2.(copy(psi_buff)), dims=(1,2)))
    norm_vec = reshape(norm, 1, 1, :)
    psi_buff = copy(psi_buff) ./ norm
    return copy(psi_buff)
end

function multislice(psi, potential, k_arr, simParams::MSA.SimulationState) 
    psi_buff = Zygote.Buffer(psi)
    psi_buff[:,:,:] = psi[:,:,:]
    Fresnel_op = MSA.build_fresnelPropagator(k_arr, simParams.λ, simParams.Δ)
    for slice_idx in 1:size(potential,3)
        trans = MSA.build_transmPropagator(potential[:,:,slice_idx], simParams.σ) 
        psi_buff = ifft(Fresnel_op .* fft(copy(psi_buff) .* trans, [1,2]), [1,2])
    end
    return fft(copy(psi_buff),[1,2])
end

function sgd_update!(model::ForwardModel, grads, η = 0.01; verbose=false)
    if verbose 
        @info("Gradients:
            ∂V=$(round.(grads.potential; digits=2))
            ∂ѱ=$(round.(grads.phase; digits=2))")
        @info("before sgd update:
        V'=$(round.(model.potential; digits=2))
        ѱ'=$(round.(model.phase; digits=2))")
    end
    model.phase .-= η .* grads.phase
    model.potential .-= η .* grads.potential
    
    if verbose
        @info("after sgd update:
        V'=$(round.(model.potential; digits=2))
        ѱ'=$(round.(model.phase; digits=2))")
    end
end
    
function optimizeDiffFM(ilr, model::ForwardModel, target_data, lr_decay=0.2, 
                        max_iter=2e2, num_logs=-1, verbose=false, sgd=false)
    opt = ADAM(ilr, (0.8, 0.99))
    @info("Running train loop")
    idx = 0
    loss_val = loss(model, target_data)
    if num_logs < 0
        num_logs = Int(min(max_iter/4, 250))
    end
    model_hist = Array{Tuple}(undef,num_logs)
    loss_hist = Array{Float32}(undef, num_logs)
    iter_hist = Array{Int}(undef, num_logs)
    hist_idx = 1
    @time while idx < max_iter && loss_val > 1e-4
        lr =  ilr * (idx + 1) .^ (-lr_decay)
        opt.eta = lr
        if mod(idx, max_iter ÷ num_logs) == 0
            loss_val = loss(model, psi2_trg, psi_k, args; verbose=verbose)
            println("lr=$(round(lr;digits=4)), Iteration=$idx, Loss=$(round(loss_val;digits=4))")
            model_hist[hist_idx] = (model.phase[:,:,1], model.potential[:,:,:])
            loss_hist[hist_idx] = loss_val
            iter_hist[hist_idx] = idx
            global hist_idx += 1
        end
        grads= Zygote.gradient(model) do m
            return loss(m, psi2_trg, psi_k, args)
        end
        grads = grads[1][]
        if sgd
            sgd_update!(model, grads, 1e-3)
        else
            Optimise.update!(opt, model.phase, grads.phase)
            Optimise.update!(opt, model.potential, grads.potential)
        end
        global idx += 1
    end
    return model_hist, loss_hist, iter_hist
end



# module ends
end