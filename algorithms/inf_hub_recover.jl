# Recover primal solution xrec to a sparse robust regression problem,
# from solution (y,β) for the perspecive dual. (Assume β>0.)
#
# This method uses the derived optimality conditions, which give the
# active indices of xrec along with an LP that we solve using Convex (Gurobi).
#
# Written by: Kellie MacPhee and Michael Friedlander
# Last updated: January 10, 2017


using Convex; cvx = Convex

function inf_hub_recover(y, β, A, b, σ, η; tol=0.01)

  (m,n) = size(A)

#   # Make sure ||y||_∞ ≤ (η/2β) ||y||_2^2
# if (norm(y, Inf) - (η/2β)*norm(y)^2 > 1e-4)
#   warn("Recovery was not possible.")
#   return (NaN, NaN)
# end

# Find inactive indices of x.
Aty = A'*y
nrmAty = norm(Aty,Inf)
xInactive = []
for i=1:n
  if  ( ( nrmAty - abs(Aty[i]) )/nrmAty > tol ) # if index i doesn't achieve ||A'*y||_∞
    xInactive = [xInactive ; i]
  end
end

# Find (nondistinguished) active gradients, and put together as columns of gradMatrix
  thresh = η/(2*β) * norm(y)^2
  gradMatrix = zeros(m,0)
  for i = 1:m
    if abs(abs(y[i]) - thresh ) < tol
      grad = zeros(m); grad[i] = sign(y[i])
      gradMatrix = [gradMatrix grad]
    end
  end
  k = size(gradMatrix, 2) # number of active gradients (not including distinguished one)

  if k==0
    warn("No active gradients.")

    λdes = (2*β^2)/(η*norm(y)^2) # what coefficient on distinguished term should be

    # Recover x by solving a simple least squares problem.
     x = cvx.Variable(n)
     λ = cvx.Variable(1)
     p = cvx.minimize( sumsquares(  (b-A*x) - (σ*λ*η/β)*y   ) + sumsquares(λ-λdes)  )

     p.constraints += ( λ == 1 )
     for i in xInactive
       p.constraints += x[i] == 0
     end

     solve!(p, GurobiSolver(OutputFlag=0))

     if ~(p.status == :Optimal)
       warn("Recovery problem not solved to optimality.")
     end

   xrec = x.value[:,1]
   return xrec

 else
  λdes = (2*β^2)/(η*norm(y)^2) # what coefficient on distinguished term should be

 # Recover x by solving a simple least squares problem.
  x = cvx.Variable(n)
  λ = cvx.Variable(1)
  μ = cvx.Variable(k) # coefficients on non-distinguished active gradients
  p = cvx.minimize( sumsquares(  (b-A*x) - (σ*λ*η/β)*y - σ*gradMatrix*μ   ) + sumsquares(λ-λdes)  )

  p.constraints += ( μ ≥ 0 )
  p.constraints += ( sum(μ) +λ == 1 )
  for i in xInactive
    p.constraints += x[i] == 0
  end

  solve!(p, GurobiSolver(OutputFlag=0))

  if ~(p.status == :Optimal)
    warn("Recovery problem not solved to optimality.")
  end

  xrec = x.value[:,1]
  return xrec

  end

end
