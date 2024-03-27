
function[obj] = mk_garch_obj(GarchType,P,O,Q)

if ~any(strcmp(GarchType,{'garch','egarch','figarch'}))
    error('Unknown GarchType.')
end

obj.P = P;
obj.O = O;
obj.Q = Q;

obj.POQ = [num2str(obj.P) num2str(obj.O) num2str(obj.Q)];

obj.GarchName = [GarchType '_' obj.POQ];

obj.GarchType = GarchType;



end