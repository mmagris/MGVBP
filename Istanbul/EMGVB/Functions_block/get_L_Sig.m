function[L,Sig] = get_L_Sig(Sig_inv,UseInvChol,J)

% Takes Sig_inv and returns Sig (inverse of Sig_inv) and the chol. factor of Sig (L)
% J is used only with UseInvChol == 1

if size(Sig_inv,1) == 1 && size(Sig_inv,2) == 1
    Sig = 1/Sig_inv;
    L = sqrt(Sig);
else

    if UseInvChol == 0
        % L is the chol factor of Sig and is computed from inv(Sig_inv).
        % L is lower-triagular.
        Sig = inv(Sig_inv);
        L = chol(Sig,'lower');

    elseif UseInvChol == 1
        % L is the chol factor of Sig.
        % L is lower-triagular.
        % L is computed from Sig_inv, without using Sig.
        % J is the anti-diagonal matrix.
        % There is one inversion (J*inv(G)*J), of the triangular matrix G
        % This gives the same L as setting.UseInvChol == 0.
        G = chol(J*Sig_inv*J,'lower');
        L = (J/G*J)';
        Sig = L*L';

    elseif UseInvChol == -1
        % L is the inverted chol factor of Sig_inv
        % (which is NOT the inverse of the chol factor is Sig!)
        % L is upper-triangular, as an upper-triangular matrix is required
        % in generating MC normal numbers from the inverse cov matrix
        L = inv(chol(Sig_inv,'upper'));
        Sig = L*L';

    end

end

% ---- This is quite different than the analogous function for MGVB! ----


% ABOUT INVERSION
% All the cases require an inversion: the alternatives are of same
% complexity.
% However,
% UseInvChol ==  0, hardest to do, inverts a full matrix
% UseInvChol == -1, inverts a triangular matrix: better than UseInvChol == 0
% UseInvChol ==  1, inverts a triangular matrix: less flops than UseInvChol == 0,
%                   yet additional multiplications involving J and L

% ABOUT L
% setting.UseInvChol == 0 and setting.UseInvChol == 1
% generate the same MC random numbers
% but setting.UseInvChol == -1 does not
% as the L from setting.UseInvChol == 0 and setting.UseInvChol == 1
% has different elements than the L from setting.UseInvChol == -1

% USAGE
% UseInvChol == -1 is the simplest and suggested. Yet its results are not
% comparable with methods that update S (e.g. MGVB). Fixing the random seed sets eps, but
% theta = mu + L*eps depends on L. L of Sig_inv is not equal to L of Sig so
% the MC draws.
% UseInvChol == 1 way to go for experiments
% UseInvChol == 0 worst as does the inversion of the full-matrix Sig_inv
% (but simple conceptualy)
end
