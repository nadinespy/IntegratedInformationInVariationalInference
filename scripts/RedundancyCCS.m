function [ R ] = RedundancyCCS(S, src1, src2, tgt, mi1, mi2, mi12)

  c = mi12 - mi1 - mi2;
  signs = [sign(mi1), sign(mi2), sign(mi12), sign(-c)];
  R = all(signs == signs(:,1), 2).*(-c);

end