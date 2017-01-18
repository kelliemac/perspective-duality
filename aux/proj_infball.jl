# Project x onto the infinity norm ball of radius t.

function proj_infball(x,t)
  n = size(x,1)

  xproj = copy(x)
  for i=1:n
    if abs(x[i]) > t
      xproj[i] = sign(x[i])*t
    end
  end

  return xproj

end
