function[] = set_fig(x,y,w,h,rev)

if nargin == 4
    rev = 0;
end

set(gcf,'units','points','position',[x,y,w,h],'renderer','painters');

if rev == 1
    set(gca, 'YDir','reverse')
end

end

% E.g. 
% set_fig(1500,400,500,300)
