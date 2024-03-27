function[setting] = setting_block(setting)

if ~isfield(setting.Block,'blks')
    error('Missing filed: setting.Block.blks')
end

blk = setting.Block.blks;
setting.Block.n = numel(blk);

n_blks = setting.Block.n;
n_par = sum(blk);


blk_indx_1 = cell(1,n_blks);
blk_indx_2 = cell(1,n_blks);
blk_indx_3 = cell(1,n_blks);
% blk_indx_4 = cell(1,n_blks);

indx            = [cumsum(blk)];
blk_indx_1{1}   = 1:indx(1);
blk_indx_2{1}   = n_par+(1:(indx(1)^2));
blk_indx_3{1}   = (1:(indx(1)^2));
% blk_indx_4{1}   = [blk_indx_1{1},blk_indx_2{1}];

for i = 2:n_blks
    blk_indx_1{i}   = (indx(i-1)+1):indx(i);
    tmp             = blk_indx_2{i-1};
    blk_indx_2{i}   = tmp(end) + (1:(blk(i)^2));
    blk_indx_3{i}   = blk_indx_2{i} - n_par;
    %     blk_indx_4{i}   = [blk_indx_1{i},blk_indx_2{i}];
end

% blk_indx            = [blk_indx_1',blk_indx_2',blk_indx_3',blk_indx_4'];

blk_indx            = [blk_indx_1',blk_indx_3',blk_indx_2'];
setting.Block.indx  = blk_indx;

% blk_indx:
% column 1: given a vector of parameters, gives the indices of the
% parameters in each block
% column 2: given Sig(:), gives the indices of the covariances in
% in each block
% column 3: given a vector of gradients Y12 = [grad(mu),grad(S(:))], gives the indices of the
% gradients wrt to the elements of grad(Sig(:)) in each block


FullPrior   = nan;
iFullPrior  = nan;

type        = zeros(n_blks,1);
NewPrior    = cell(n_blks,1);
iNewPrior   = cell(n_blks,1);

setting.FullPrior   = nan;
setting.iFullPrior  = nan;

% If user specifies one scalar/vector/matrix instead of cells
if ~iscell(setting.Prior.Sig)

    % NewPrior replaces the non-cell prior specifiation with a new version of the prior
    % divided in blocks, designed as the postrior.
    NewPrior = cell(n_blks,1);

    OldPrior = setting.Prior.Sig;
    if size(OldPrior,2)>size(OldPrior,1)
        OldPrior = OldPrior';
    end

    % If is a scalar
    if size(OldPrior,1) == 1 && size(OldPrior,2) == 1

        for b = 1:n_blks % split this vector across the posterior blocks
            indx = blk_indx{b,1};
            NewPrior{b} = OldPrior(1);
            type(b)     = get_type(NewPrior{b},indx);
            iNewPrior{b} = get_inverse(NewPrior{b},type(b));
        end

        % If is a vector
    elseif size(OldPrior,1) == n_par && size(OldPrior,2) == 1

        for b = 1:n_blks % replicate this scalar across the posterior blocks
            indx        = blk_indx{b,1};
            NewPrior{b} = OldPrior(indx,1);
            type(b)     = get_type(NewPrior{b},blk(b));
            iNewPrior{b} = get_inverse(NewPrior{b},type(b));
        end

        % If is a matrix (of size n_par x n_par)
    elseif size(OldPrior,1) == n_par && size(OldPrior,2) == n_par

        for b = 1:n_blks %split this matrix across the posterior blocks
            indx        = blk_indx{b,1};
            NewPrior{b} = OldPrior(indx,indx);
            type(b)     = get_type(NewPrior{b},blk(b));
        end

        % Check if the specified prior matrix matches the block-diagonal
        % form of the posterior.
        if any(any(blkdiag(NewPrior{:}) ~= OldPrior))
            % If not, invert the full matrix and extract diagonal blocks
            FullPrior           = OldPrior;
            iFullPrior          = inv(OldPrior);
            for b = 1:n_blks
                indx            = blk_indx{b,1};
                iNewPrior{b}    = iFullPrior(indx,indx);
            end
        else
            % If yes, invert the blocks
            for b = 1:n_blks
                iNewPrior{b} = get_inverse(NewPrior{b},type(b));
            end

        end


    else
        error('Check dimensions of the prior specification.')
    end

    % Used in gradient comptutations
    setting.Prior.Sig = NewPrior;
    setting.Sig_inv_0 = iNewPrior;
    setting.Sig0_type = type;

    % Save the specified prior to use in h-function computation
    setting.FullPrior = FullPrior;
    setting.iFullPrior = iFullPrior;


else

    % If multiple cells are specified for the prior, check that thery are
    % coherent with the blocks of the posterior
    if size(setting.Prior.Sig) ~= n_blks
        error('The number of prior blocks does not match the number of posterior blocks.')
    end

    % make sure that prior blocks contain are scalars, matrices or COLUMN vectors
    for b = 1:n_blks
        NewPrior{b} = setting.Prior.Sig{b};
        if size(NewPrior{b},2)> size(NewPrior{b},1)
            NewPrior{b} = NewPrior{b}';
        end
        type(b) = get_type(NewPrior{b},blk(b));
    end

    % For each prior block compute its inverse and save its type.
    for b = 1:n_blks
        iNewPrior{b} = get_inverse(NewPrior{b},type(b));
    end

    % Store NewPriorand iNewPrior
    setting.Prior.Sig = NewPrior;
    setting.Sig_inv_0 = iNewPrior;
    setting.Sig0_type = type;

end

end



function[type] = get_type(Sb,blk_size)

if size(Sb,1) == 1 && size(Sb,2) == 1
    type = 3;
elseif size(Sb,1) == blk_size && size(Sb,2) == 1
    type = 2;
elseif size(Sb,1) == blk_size && size(Sb,2) == blk_size
    type = 1;
else
    error('Size of the prior block does not match the size of the posterior block.')
end

end


function[iS] = get_inverse(S,type)
if type == 1
    try
        chol(S);
    catch ME
        error('Prior matrix not symmetric positive definite')
    end
    iS = inv(S);
elseif type == 2
    iS = 1./S;
elseif type == 3
    iS = 1/S;
else
    error('Unknown type.')
end
end
