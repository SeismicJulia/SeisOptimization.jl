"""
   solving ||Ax-b|| via CGLS, This code is a translated version of SOL's matlab code
   the resNE (normalized relative error is ||A'b-A'Ax||/||A'b|| )
"""
function cgls(A, b::Vector{Tv}; params=Dict(),
              x0=0.0, shift=0.0, tol=1e-6, maxIter=30, print_flag=false) where {Tv<:Number}

    # determine the type of A
    if isa(A, Matrix)
       explicitA = true
    elseif isa(A, Function)
       explicitA = false
    else
       error("A must be a matrix or a function")
    end

    # initial guess of the unknowns
    if explicitA

       (m, n) = size(A)

       # provided initial guess
       if x0 != 0.0
          x = copy(x0)
          r = b - A * x
          s = A' * r - shift * x
       # zero vector as initial guess
       else
          x = zeros(eltype(b), n)
          r = copy(b)
          s = A' * r
       end

    else

       m = length(b)

       # provided initial guess
       if x0 != 0.0
          x = copy(x0)
          r = b - A(x, 1; params...)
          s = A(r, 2; params...) - shift * x
          n = length(s)
       # zero vector as initial guess
       else
          r = copy(b)
          s = A(b, 2; params...)
          n = length(s)
          x = zeros(eltype(b), n)
       end

    end

    data_fitting = dot(r,r)
    constraint   = 0.0
    cost0 = data_fitting + constraint

    # initialize some intermidiate vectors
    p = copy(s)
    norms0= vecnorm(s)
    gamma = norms0^2
    normx = vecnorm(x)
    xmax  = normx     # keep the initial one
    resNE = 1.0

    gamma0= copy(gamma)
    delta = 0.0

    # iteration counter and stop condition
    k = 0
    run_flag = true

    if print_flag
       header = "  k         data_fitting           constraint             normx                resNE"
       println(""); println(header);
       @printf("%3.0f %20.10e %20.10e %20.10e %20.10e\n", k, data_fitting, constraint, normx, resNE);
    end

    while (k < maxIter) && run_flag

          k = k + 1

          if explicitA
             q = A * p
          else
             q = A(p, 1; params...)
          end

          delta = (vecnorm(q))^2 + shift * (vecnorm(p))^2
          indefinite = delta <= 0 ? true : false
          delta      = delta == 0.? eps(): delta

          alpha = gamma / delta

          x = x + alpha * p
          r = r - alpha * q

          data_fitting = dot(r,r)
          constraint   = shift * dot(x, x)
          cost = data_fitting + constraint

          if explicitA
             s = A' * r - shift * x
          else
             s = A(r, 2; params...) - shift * x
          end

          norms  = norm(s)
          gamma0 = copy(gamma)
          gamma  = norms^2
          beta   = gamma / gamma0

          p = s + beta * p

          # check the stopping crietia
          normx = norm(x)
          xmax  = normx > xmax ? normx : xmax
          if norms <= norms0 * tol || normx * tol >= 1.0
             run_flag = false
          end

          # print information
          # resNE = norms / norms0
          resNE = cost  / cost0
          if print_flag
             @printf("%3.0f %20.10e %20.10e %20.10e %20.10e\n", k, data_fitting, constraint, normx, resNE);
          end

    end

    return x, resNE

end

# ==============================================================================
#                 CGLS based on hard drive
# ==============================================================================
"""
   solving ||Ax-b|| via CGLS
"""
function cgls(A::Function, b::Ts; x0="NULL", path="NULL", params=[],
         shift=0.0, tol=1e-6, maxIter=50, print_flag=false) where {Ts<:Union{String, Vector{String}}}

    if path == "NULL"
       path = pwd()
    end

    # create folder to save files of model size
    path_model = joinpath(path, "model")
    if !isdir(path_model)
       mkdir(path_model)
       if !isdir(path_model)
          error("could not create directory")
       end
    end
    x = join([path_model "/x.bin"])
    s = join([path_model "/s.bin"])
    p = join([path_model "/p.bin"])

    # create folders to save residue
    path_residue   = joinpath(path, "residue")
    if !isdir(path_residue)
       mkdir(path_residue)
       if !isdir(path_residue)
          error("could not create directory")
       end
    end

    # create folders to save synthetic data with current model
    path_forward = joinpath(path, "forward")
    if !isdir(path_forward)
       mkdir(path_forward)
       if !isdir(path_forward)
          error("could not create directory")
       end
    end

    # intermidiate variables
    if typeof(b) == String
       r = join([path_residue "/res.bin"])
       q = join([path_forward "/fwd.bin"])
    elseif typeof(b) == Vector{String}
       n = length(b)
       r = Vector{String}(n)
       q = Vector{String}(n)
       for i = 1 : n
           r[i] = join([path_residue "/res" "_" "$i" ".bin"])
           q[i] = join([path_forward "/fwd" "_" "$i" ".bin"])
       end
    end

    # provided initial guess
    if x0 != "NULL"

       cp(x0, x, remove_destination=true)
       dcal = A(x, 1; params=params)
       x_plus_alpha_q!(r, b, -1.0, dcal)

       if typeof(params) <: Dict{Symbol, Any}
          params[:path_m] = s
       elseif typeof(params) == Vector{Dict}
         for i = 1 : length(params)
             params[i][:path_m] = s
         end
       end
       s = A(r, 2; params=params)
       x_plus_alpha_q!(s, -shift, x)

    # zero vector as initial guess
    else

       cp(b, r, remove_destination=true)
       if typeof(params) <: Dict{Symbol, Any}
          params[:path_m] = s
       elseif typeof(params) == Vector{Dict}
          for i = 1 : length(params)
              params[i][:path_m] = s
          end
       end
       s = A(r, 2; params=params)

       # create x
       (hdr, ds) = read_USdata(s)
       tmp = zeros(ds)
       write_USdata(x, hdr, tmp)
    end

    # compute residue
    data_fitting = (norm(r))^2
    constraint   = 0.0
    cost0 = data_fitting + constraint
    convergence = Float64[]; push!(convergence, 1.0);

    # initialize some intermidiate vectors
    cp(s, p, remove_destination=true)

    norms0= norm(s)
    gamma = norms0^2
    normx = norm(x)
    xmax  = normx     # keep the initial one
    resNE = 1.0

    gamma0= copy(gamma)
    delta = 0.0

    # iteration counter and stop condition
    k = 0
    run_flag = true

    if print_flag
       header = "  k         data_fitting           constraint             normx                resNE"
       println(""); println(header);
       @printf("%3.0f %20.10e %20.10e %20.10e %20.10e\n", k, data_fitting, constraint, normx, resNE);
    end

    if typeof(params) <: Dict{Symbol, Any}
       params[:path_fwd] = q
    elseif typeof(params) == Vector{Dict}
       for i = 1 : length(params)
           params[i][:path_fwd] = q[i]
       end
    end

    while (k < maxIter) && run_flag

          k = k + 1

          q = A(p, 1; params=params)

          delta = (norm(q))^2 + shift * (norm(p))^2
          indefinite = delta <= 0 ? true : false
          delta      = delta == 0.? eps(): delta

          alpha = gamma / delta

          x_plus_alpha_q!(x,  alpha, p)
          x_plus_alpha_q!(r, -alpha, q)

          data_fitting = (norm(r))^2
          constraint   = shift * (norm(x))^2
          cost = data_fitting + constraint

          # save the intermidiate result
          path_iter = join([path_model "/iteration" "_" "$k" ".bin"])
          cp(x, path_iter, remove_destination=true)

          s = A(r, 2; params=params)
          x_plus_alpha_q!(s, -shift, x)

          norms  = norm(s)
          gamma0 = gamma
          gamma  = norms^2
          beta   = gamma / gamma0

          (hdr, ds) = read_USdata(s)
          (hdr, dp) = read_USdata(p)
          dp = ds + beta * dp
          write_USdata(p, hdr, dp)

          # check the stopping crietia
          normx = norm(x)
          xmax  = normx > xmax ? normx : xmax
          if norms <= norms0 * tol || normx * tol >= 1.0
             run_flag = false
          end

          # print information
          resNE = cost / cost0
          if print_flag
             @printf("%3.0f %20.10e %20.10e %20.10e %20.10e\n", k, data_fitting, constraint, normx, resNE);
          end
          push!(convergence, resNE);
    end

    return x, convergence

end
