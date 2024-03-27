function[] = print_training_info(verbose,stop,iter,window_size,LB,LB_smooth)
if verbose == 1
    if iter> window_size

        LBimporved = LB_smooth(iter-window_size)>=max(LB_smooth(1:iter-window_size));

        if LBimporved
            str = '*';
        else
            str = ' ';
        end

        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter-window_size)),str])
    else
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
    end
end

if verbose == 2 && stop == true
    if iter> window_size
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter-window_size))])
    else
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
    end
end
end
