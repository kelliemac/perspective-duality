# General Chambolle-Pock method.
# Written by: Kellie MacPhee and Michael Friedlander
# Last updated: December 16, 2016
#
# Solves the problem:     minimize  F(Kx) + G(x)
#
# via the iteration:            ynew = prox_{σF*} (y + σ ⋅ K * xb)
#                                      xnew = prox_{τG} (x + τ ⋅ K' * y)
#                                      xb = xnew + Θ(xnew - x)
#
# Note: prox functions should have form
#     x' = proxG(x,t) = inf_{z} ( G(y) + (0.5/t)⋅||z-x||₂^2 )
#     y' = proxFc(y,t) = inf_{z} ( F*(z) + (0.5/t)⋅||z-y||₂^2 )

function chamb_pock( proxFc, proxG, K, x0, y0;  Θ=1, damp=0.99, Gsteplengthen=1, maxIts=50*maximum(size(K)), tol=1e-8, verbose=true)

  m, n = size(K)

  # Set step sizes τ,σ to ensure that τσ⋅||K||^2 < 1.
  L = norm(K)
  τ = (damp*Gsteplengthen)/L # larger value of Gsteplength increases step size τ corr. to proxG step
  σ = damp/(L*Gsteplengthen)

  # Initialize.
  x = copy(x0); xb = copy(x0); y = copy(y0)
  pr_iterates = collect(xb); du_iterates = collect(y) # iterates are columns
 steps = [] # col vector of normalized step sizes (convergence criterion)
 if verbose == true
   @printf("%4s  %10s\n","itn","normalized step size")
 end

  # Run Chambolle-Pock on the problem.
  for i=1:maxIts
    ynew = proxFc( y + σ*(K*xb), σ )
    xnew = proxG( x - τ*(K'*ynew), τ )
    xb = xnew + Θ*(xnew- x)

    # Compute and print normalized step length Δ.
    Δ = (norm(xnew-x)/τ) + (norm(ynew-y)/σ)
    if verbose == true
      @printf("%4i  %10.3e\n", i, Δ)
    end

    # Check convergence criterion.
    if Δ ≤ tol
      break
    end

    # Record and update iterates.
    x = copy(xnew)
    y = copy(ynew)
    pr_iterates = [pr_iterates x]
    du_iterates = [du_iterates y]
    steps = [steps; Δ]

  end

  # Return all primal and dual  iterates, and the normalized step sizes.
  return (pr_iterates, du_iterates, steps)

end
