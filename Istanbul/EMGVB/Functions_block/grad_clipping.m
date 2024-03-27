function Y12 = grad_clipping(Y12,max_grad)
if max_grad>0
    grad_norm = norm(Y12);
    norm_gradient_threshold = max_grad;
    if grad_norm > norm_gradient_threshold
        Y12 = (norm_gradient_threshold/grad_norm)*Y12;
    end
end
end
