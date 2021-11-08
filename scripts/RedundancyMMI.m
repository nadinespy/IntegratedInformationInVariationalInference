function [ R ] = RedundancyMMI(bX, src1, src2, tgt, mi1, mi2, mi12)
  if mean(mi1) < mean(mi2)
    R = mi1;
  else
    R = mi2;
  end
end