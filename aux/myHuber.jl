# Evaluate huber function with paramter η, at x.

function myHuber(x, η)

  hub = 0
  for i = 1:size(x,1)
    if abs(x[i]) ≤ η
      hub += (0.5/η)*(x[i]^2)
    else
      hub += abs(x[i]) -η/2
    end
  end

  return hub

end
