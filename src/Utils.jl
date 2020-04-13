using Plots

function comp_plot(psi_out, psi_in, V)
    l = @layout [a b c]
    p1 = heatmap(abs2.(psi_in[:,:,1]) .^ 0.25, aspect_ratio=1, framestyle = :none)
    p2 = heatmap(psi_out[:,:,1] .^ 0.1, aspect_ratio=1, framestyle = :none)
    p3 = heatmap(sum(V, dims=3)[:,:,1], aspect_ratio=1, framestyle = :none, color = :vibrant_grad)
    heatmap(p1, p2, p3, layout=l, title=["Psi_in" "Data" "V"], size=(1200,400))
    
end

function V_plot(V_true, V_pred)
    l = @layout [a b]
    p1 = heatmap(sum(real.(V_true), dims=3)[:,:,1], aspect_ratio=1, framestyle = :none)
    p2 = heatmap(sum(real.(V_pred), dims=3)[:,:,1], aspect_ratio=1, framestyle = :none)
    heatmap(p1, p2, layout=l , title=["V-target" "V-predicted"], size=(800,400))
end
        
function grad_plot(grads)
    l = @layout [a b]
    p1 = heatmap(angle.(grads.psi[:,:,1]), aspect_ratio=1, framestyle = :none)
    p2 = heatmap(angle.(grads.potential)[:,:,1], aspect_ratio=1, framestyle = :none)
    heatmap(p1, p2, layout=l, title=["arg(Grad_Psi)" "arg(Grad_V)"], size=(800,400))
end
    
function diff_plot(psi_out_pred, psi_out_true;clims=:auto)
    pred = sum(psi_out_pred;dims=3)[:,:,1]
    gtruth = sum(psi_out_true;dims=3)[:,:,1]
    heatmap((pred .- gtruth) ./ gtruth .* 100, 
        aspect_ratio=1, framestyle= :none, title="% Difference", clims= clims)
end

function build_psi(amp, phase)
    psi_k = amp .* exp.(-1.0f0 * im .* phase)
    psi_x = ifft(psi_k, [1,2])
    psi_x .= fftshift(psi_x, [1,2])
    norms = sqrt.(sum(abs2, psi_x, dims=[1,2]))
    psi_x ./= norms
    return psi_x
end
    
function get_history(model_hist, iter_hist)
    anim = @animate for ((psi,potential), iter) in zip(model_hist, iter_hist)
        psi = real.(psi[:,:,1]) 
        pot = sum(real.(potential), dims=3)[:,:,1]
        l = @layout [a b]
        p1 = heatmap(psi, aspect_ratio=1, framestyle = :none, color= :vibrant_grad)
        p2 = heatmap(pot , aspect_ratio=1, framestyle = :none, color= :vibrant_grad)
        heatmap(p1, p2, layout=l, title=["Psi_in\n Iterations=$iter" "V"], size=(800,400))
        end
    return anim
end

