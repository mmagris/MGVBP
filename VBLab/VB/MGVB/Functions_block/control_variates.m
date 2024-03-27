function[c] = control_variates(A,B,S,do)
if do
    c = (mean(A.*B)-mean(A).*mean(B))./var(B)*S/(S-1);
else
    c = 0;
end
end

