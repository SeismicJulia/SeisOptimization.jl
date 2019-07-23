const FloatOrComplex = Union{Float32, Float64, ComplexF32, ComplexF64}

"""
   define function as linear operator
"""
struct LinearOperator{T<:DataType, Ti<:Int64, F1, F2}

    data_format       :: T

    # forward operator
    forward_operator  :: F1
    forward_params    :: Union{Nothing, NamedTuple}
    is_forward_mutate :: Bool

    # adjoint operator
    adjoint_operator  :: F2
    adjoint_params    :: Union{Nothing, NamedTuple}
    is_adjoint_mutate :: Bool

    # size of the linear operator
    m                 :: Ti
    n                 :: Ti

    # some properties
    is_symmetrical    :: Bool
    is_hermitian      :: Bool
end

"""
   Constructor for linear operator which consist of forward and its adjoint operators
"""
function LinearOperator(data_format::DataType, forward_op::F1, adjoint_op::F2, m::Ti, n::Ti;
                        forward_params=nothing, adjoint_params=nothing)  where {Ti<:Int64, F1, F2}

    # check data fromat
    data_format <: FloatOrComplex || error("only support float or complex")

    # forward operator mutating or not
    nargs = first(methods(forward_op)).nargs
    (nargs == 2 || nargs == 3) || error("wrong number of input to forward operator")
    is_forward_mutate =  (nargs == 3)

    # adjoint operator mutating or not
    nargs = first(methods(adjoint_op)).nargs
    (nargs == 2 || nargs == 3) || error("wrong number of input to adjoint operator")
    is_adjoint_mutate =  (nargs == 3)

    # proerties
    is_symmetrical = false
    is_hermitian   = false

    return LinearOperator(data_format,
                          forward_op, forward_params, is_forward_mutate,
                          adjoint_op, adjoint_params, is_adjoint_mutate,
                          m, n,
                          is_symmetrical, is_hermitian)
end

"""
   Constructor for symmetrical or hermitian linear operator
"""
function LinearOperator(data_format::DataType, forward_op::F1, m::Ti; forward_params=nothing) where {Ti<:Int64, F1, F2}

    # check data fromat
    data_format <: FloatOrComplex || error("only support float or complex")

    # operator support in-place operation
    is_forward_mutate = first(methods(forward_op)).nargs == 3

    # adjoint is same as forward operator
    adjoint_op     = forward_op
    adjoint_params = forward_params
    is_adjoint_mutate = is_forward_mutate
    n = m

    # proerties
    if data_format <: AbstractFloat
       is_symmetrical = true
       is_hermitian   = false
    else
       is_symmetrical = false
       is_hermitian   = true
    end


    return LinearOperator(data_format,
                          forward_op, forward_params, is_forward_mutate,
                          adjoint_op, adjoint_params, is_adjoint_mutate,
                          m, n,
                          is_symmetrical, is_hermitian)
end

"""
   Customerize printing function for linear operator
"""
function Base.show(io::IO, L::LinearOperator)
    @printf("data_format = %10s\n", L.data_format)
    @printf("forward operator = %20s, ismutating = %6s\n", L.forward_operator, L.is_forward_mutate)
    @printf("adjoint operator = %20s, ismutating = %6s\n", L.adjoint_operator, L.is_adjoint_mutate)
    @printf("size of linear operator = (%d, %d)\n", L.m, L.n)
    @printf("issymmetrical = %6s, ishermitian = %6s\n", L.is_symmetrical, L.is_hermitian)
    return nothing
end

# enquire the properties of linear operator
Base.eltype(L::LinearOperator) = L.data_format
Base.ndims(L::LinearOperator)  = 2
Base.size(L::LinearOperator)   = (L.m, L.n)
Base.size(L::LinearOperator, n::Integer) = (n==1 || n==2) ? size(L)[n] : error("Linear operator only have two dimensions")
Base.isreal(L::LinearOperator) = L.data_format <: AbstractFloat
LinearAlgebra.issymmetric(L::LinearOperator) = L.is_symmetrical
LinearAlgebra.ishermitian(L::LinearOperator) = L.is_hermitian
ismutating(L::LinearOperator) = (L.is_forward_mutate, L.is_adjoint_mutate)

"""
   Apply the linear operator to an Abstract vector
"""
function A_mul_b!(y::AbstractVector, L::LinearOperator, x::AbstractVector)

    # check the length of input and output
    (length(x) == L.n && length(y) == L.m) || throw(DimensionMismatch("check the length of x or y"))

    # apply forward operator
    if ismutating(L)[1]

       L.forward_params == nothing ? L.forward_operator(y, x) : L.forward_operator(y, x; L.forward_params...)

    else

       L.forward_params == nothing ? copyto!(y,L.forward_operator(x)) : copyto!(y,L.forward_operator(x; L.forward_params...))
    end

    return nothing
end

"""
   define the multiplication operator
"""
function Base.:(*)(L::LinearOperator, x::AbstractVector)

    # check the length of input
    length(x) == L.n || throw(DimensionMismatch())

    if ismutating(L)[1]

       # allocate memory
       y = similar(x, promote_type(eltype(L), eltype(x)), L.m)
       L.forward_params == nothing ? L.forward_operator(y, x) : L.forward_operator(y, x; L.forward_params...)

    else

       L.forward_params == nothing ? y = L.forward_operator(x) : y = L.forward_operator(x; L.forward_params...)

    end

    return y
end

"""
   apply the adjoint operator
"""
function Ac_mul_b!(y::AbstractVector, L::LinearOperator, x::AbstractVector)

    # check the length of input and output
    (length(x) == L.m && length(y) == L.n) || throw(DimensionMismatch("check the length of x or y"))

    # apply adjoint operator
    if ismutating(L)[2]

       L.adjoint_params == nothing ? L.adjoint_operator(y, x) : L.adjoint_operator(y, x; L.adjoint_params...)

    else

       L.adjoint_params == nothing ? copyto!(y,L.adjoint_operator(x)) : copyto!(y,L.adjoint_operator(x; L.adjoint_params...))
    end

    return nothing
end


"""
   apply the adjoint operator, At_mulb! is equal to Ac_mul_b! when the linear operator is real
"""
function At_mul_b!(y::AbstractVector, L::LinearOperator, x::AbstractVector)

    # check the length of input and output
    (length(x) == L.m && length(y) == L.n) || throw(DimensionMismatch("check the length of x or y"))


    if !isreal(L)
       x = conj(x)
    end

    # apply adjoint operator
    if ismutating(L)[2]

       L.adjoint_params == nothing ? L.adjoint_operator(y, x) : L.adjoint_operator(y, x; L.adjoint_params...)

    else

       L.adjoint_params == nothing ? copyto!(y,L.adjoint_operator(x)) : copyto!(y,L.adjoint_operator(x; L.adjoint_params...))
    end

    if !isreal(L)
       conj!(y)
    end

    return nothing
end

"""
   get the corresponding matrix of a linear operator
"""
function Base.Matrix(L::LinearOperator)

    # size of linear operator
    (m, n) = size(L)

    # allocate space
    A = Matrix{L.data_format}(undef, L.m, L.n)

    # input vector
    x = zeros(L.data_format, n)

    # probing column by column
    @inbounds for i = 1 : n
         x[i] = one(L.data_format)
         A_mul_b!(view(A,:,i), L, x)

         # prepare for next column
         x[i] = zero(L.data_format)
    end

    return A
end

"""
   get the corresponding sparse matrix of a linear operator
"""
function SparseArrays.sparse(L::LinearOperator)

    # size of linear operator
    (m, n) = size(L)

    # data format
    Tv = eltype(L)

    # the variable for sparse matrix
    rowval = Int64[]
    colptr = zeros(Int64, n+1)
    nzval  = Tv[]

    # vector input
    x = zeros(Tv, n)
    y = Vector{Tv}(undef, m)

    # loop over all columns
    @inbounds for i = 1 : n
        x[i] = one(Tv)
        A_mul_b!(y, L, x)
        js = findall(!iszero, y)
        colptr[i] = length(nzval) + 1

        if length(js) > 0
           append!(rowval, js)
           append!(nzval, y[js])
        end

        x[i] = zero(Tv)
    end
    colptr[n+1] = length(nzval) + 1

    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end
