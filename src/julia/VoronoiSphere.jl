macro fields(expr)
    @assert expr.head == :(::)
    typ = expr.args[2]
    lines = [:($field::$typ) for field in expr.args[1].args]
    return esc(Expr(:block, lines...))
end

struct VoronoiSphere{
    F<:AbstractFloat,
    VI<:AbstractVector{Int32},  # Vectors of integers
    VR<:AbstractVector{F},      # Vectors of reals
    VI2, VI3, VR3, VI4, VR4, VI6, VR6, VI10, VR10,
    AR<:AbstractArray{F,3},     # 3D array of reals
    MP<:AbstractMatrix{NTuple{3,F}}, # Matrices of 3D points
} <: UnstructuredDomain
    @fields (primal_num, dual_num, edge_num)::Int32
    @fields (primal_deg, dual_deg, trisk_deg)::VI
    @fields (Ai, lon_i, lat_i, Av, lon_v, lat_v)::VR
    @fields (le, de, le_de, lon_e, lat_e, angle_e)::VR
    @fields (edge_left_right, edge_down_up)::VI2
    @fields (dual_edge, dual_vertex)::VI3
    @fields (Avi, Riv2, dual_ne, dual_bounds_lon, dual_bounds_lat)::VR3
    edge_kite ::VI4
    edge_perp :: VR4
    @fields (primal_edge, primal_vertex, primal_neighbour)::VI6
    @fields (primal_ne, Aiv, primal_bounds_lon, primal_bounds_lat)::VR6
    trisk :: VI10
    wee :: VR10
    primal_perot_cov::AR
    primal_grad3d::MP
    # computed
    inv_Ai::VR
    @fields (xyz_i, elon_i, elat_i)::VR3
    @fields (xyz_e, elon_e, elat_e, normal_e, tangent_e)::VR3
    @fields (xyz_v, elon_v, elat_v)::VR3
    @fields (cen2edge, cen2vertex)::MP
end
const VSph = VoronoiSphere

Base.show(io::IO, ::Type{<:VoronoiSphere{F}}) where {F} = print(io, "VoronoiSphere{$F}")
Base.show(io::IO, sphere::VoronoiSphere) =
    print(io, "VoronoiSphere($(length(sphere.Ai)) cells, $(length(sphere.Av)) dual cells)")

struct StructDict
    dict::Dict{Symbol, Any}
    StructDict(itr) = new(Dict(itr))
end
Base.getindex(sd::StructDict, sym::Symbol) = getindex(sd.dict, sym)
Base.setindex!(sd::StructDict, val, sym::Symbol) = setindex!(sd.dict, val, sym)
Base.getproperty(sd::StructDict, sym::Symbol) = (sym==:dict) ? getfield(sd, sym) : sd.dict[sym]
Base.setproperty!(sd::StructDict, sym::Symbol, val) = setindex!(sd.dict, val, sym)

function VoronoiSphere(read_data::Function; prec = Float32)
    # read Float data from file ; convert to `prec`
    real_names = (
        (:primal_ne, :dual_ne)...,
        (:Ai, :lon_i, :lat_i, :Av, :lon_v, :lat_v)...,
        (:le, :de, :le_de, :lon_e, :lat_e, :angle_e)...,
        (:primal_bounds_lon, :primal_bounds_lat, :dual_bounds_lon, :dual_bounds_lat)...,
        (:Riv2, :Aiv, :Avi, :wee, :primal_perot_cov, :edge_perp, :primal_grad3d)...,
    )
    data = StructDict(name => prec.(read_data(name)) for name in real_names)

    # read int data from file; convert to Int32
    int_names = (
        (:primal_num, :dual_num, :edge_num)...,
        (:primal_deg, :dual_deg, :trisk_deg)...,
        (:primal_edge, :primal_vertex, :primal_neighbour, :dual_edge, :dual_vertex)...,
        (:edge_left_right, :edge_down_up, :trisk, :edge_kite, )...,
    )

    for name in int_names
       setproperty!(data, name, Int32.(read_data(name)))
    end
    nums = (data.primal_num, data.dual_num, data.edge_num)

    # Convert 3D vectors to tuples
    vec2tup(x) = [ map(dim->x[i,j,dim], (1,2,3)) for i in axes(x, 1), j in axes(x, 2)]
    data.primal_grad3d = vec2tup(data.primal_grad3d)

    # Extra stuff which can be computed without worrying about halos
    data.inv_Ai = inv.(data.Ai)
    data.xyz_i, data.elon_i, data.elat_i = local_bases(data.lon_i, data.lat_i)
    data.xyz_e, data.elon_e, data.elat_e = local_bases(data.lon_e, data.lat_e)
    data.xyz_v, data.elon_v, data.elat_v = local_bases(data.lon_v, data.lat_v)
    data.normal_e, data.tangent_e =
        normal_tangents(data.elon_e, data.elat_e, data.angle_e)
    data.cen2edge = center_to_edge(data.xyz_i, data.xyz_e, data.primal_deg, data.primal_edge)
    data.cen2vertex = center_to_edge(data.xyz_i, data.xyz_v, data.primal_deg, data.primal_vertex)

    # Convert 2D arrays into vectors of tuples
    for name in (:dual_edge, :dual_vertex, :edge_left_right, :edge_down_up, :edge_kite, :Avi, :Riv2, :dual_ne, :dual_bounds_lon, :dual_bounds_lat)
        data[name] = vector_of_tuples(data[name])
    end
    for name in (:primal_edge, :primal_vertex, :primal_neighbour, :primal_ne, :Aiv, :primal_bounds_lat, :primal_bounds_lon)
        data[name] = vector_of_tuples(data[name], data.primal_deg)
    end
    for name in (:trisk, :wee)
        data[name] = vector_of_tuples(data[name], data.trisk_deg)
    end

    # Store everything into a VoronoiSphere object
    return VoronoiSphere((crop(nums, data[name], name) for name in fieldnames(VoronoiSphere))...)
end

local_bases(lons, lats) = @. zipper = local_basis(lons, lats)
function local_basis(lon, lat)
    sinlon, coslon = sincos(lon)
    sinlat, coslat = sincos(lat)
    return (
        (coslon * coslat, sinlon * coslat, sinlat),
        (-sinlon, coslon, zero(sinlat)),
        (-coslon * sinlat, -sinlon * sinlat, coslat),
    )
end

normal_tangents(elons, elats, angles) = @. zipper = normal_tangent(elons, elats, angles)
function normal_tangent(elon, elat, angle)
    sina, cosa = sincos(angle)
    return (@. cosa * elon + sina * elat), (@. cosa * elat - sina * elon)
end

function center_to_edge(xyz_i, xyz_e, primal_degree, primal_edge)
    dxyz = similar(primal_edge, eltype(xyz_i))
    for cell in eachindex(xyz_i)
        center = xyz_i[cell]
        for edge in 1:primal_degree[cell]
            dxyz[edge, cell] = xyz_e[primal_edge[edge, cell]] .- center
        end
    end
    return dxyz
end

@inline Base.eltype(dom::VSph) = eltype(dom.Ai)

@inline primal(dom::VSph) = SubMesh{:scalar,typeof(dom)}(dom)

vector_of_tuples(data::AbstractMatrix) = [ntuple(i->data[i,j], size(data,1)) for j in axes(data,2)]

function vector_of_tuples(data::AbstractMatrix{<:Integer}, degree)
    for j in eachindex(degree)
        deg = degree[j]
        for i in deg+1:size(data,1)
            data[i,j] = data[deg,j]
        end
    end
    return vector_of_tuples(data)
end
function vector_of_tuples(data::AbstractMatrix{<:AbstractFloat}, degree)
    for i in eachindex(degree)
        deg = degree[i]
        for j in deg+1:size(data,1)
            data[j,i] = 0
        end
    end
    return vector_of_tuples(data)
end

function crop((primal_num, dual_num, edge_num), data, name::Symbol)
    if name in (
        :primal_deg,
        :primal_edge,
        :primal_vertex,
        :primal_ne,
        :Ai,
        :lon_i,
        :lat_i,
        :primal_bounds_lon,
        :primal_bounds_lat,
    )
        num = primal_num
    elseif name in (
        :dual_deg,
        :dual_edge,
        :dual_vertex,
        :dual_ne,
        :Av,
        :lon_v,
        :lat_v,
        :dual_bounds_lon,
        :dual_bounds_lat,
        :Riv2,
    )
        num = dual_num
    elseif name in (
        :trisk_deg,
        :edge_left_right,
        :edge_down_up,
        :trisk,
        :le,
        :de,
        :le_de,
        :lon_e,
        :lat_e,
        :angle_e,
        :wee,
    )
        num = edge_num
    elseif name == :primal_perot_cov
        return data[:, 1:primal_num, :]
    else
        return data
    end
    if isa(data, AbstractVector)
        return data[1:num]
    else
        return data[:, 1:num]
    end
end

#====================== Allocate ======================#

array(::Nothing, dom::VSph, F, dims...) = similar(dom.Ai, F, dims...)

allocate_field(::Val{:scalar}, dom::VSph, F::Type{<:Real}, backend = nothing) =
    array(backend, dom, F, length(dom.Ai))
allocate_field(::Val{:dual}, dom::VSph, F::Type{<:Real}, backend = nothing) =
    array(backend, dom, F, length(dom.Av))
allocate_field(::Val{:vector}, dom::VSph, F::Type{<:Real}, backend = nothing) =
    array(backend, dom, F, length(dom.le))

allocate_shell(::Val{:scalar}, dom::VSph, nz, F::Type, backend = nothing) =
    array(backend, dom, F, nz, length(dom.Ai))
allocate_shell(::Val{:dual}, dom::VSph, nz, F::Type, backend = nothing) =
    array(backend, dom, F, nz, length(dom.Av))
allocate_shell(::Val{:vector}, dom::VSph, nz, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.le))
allocate_shell(::Val{:scalar}, dom::VSph, nz, nq, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.Ai), nq)
allocate_shell(::Val{:dual}, dom::VSph, nz, nq, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.Av), nq)
allocate_shell(::Val{:vector}, dom::VSph, nz, nq, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.le), nq)

@inline periodize!(data, ::Shell{Nz,<:VSph}, backend) where {Nz} = data
@inline periodize!(data, ::Shell{Nz,<:VSph}) where {Nz} = data
@inline periodize!(datas::Tuple, ::Shell{Nz,<:VSph}, args...) where {Nz} = datas

#====================== Effective resolution ======================#

normL2(f) = sqrt(sum(x -> x^2, f) / length(f))

"""
Estimates the largest eigenvalue `-lambda=dx^-2` of the scalar Laplace operator and returns `dx`
which is a (non-dimensional) length on the unit sphere characterizing the mesh resolution.
By design, the Courant number for the wave equation with unit wave speed solved with time step `dt` is `dt/dx`.
"""
function laplace_dx(mesh::VoronoiSphere, mgr = nothing)
    rng = MersenneTwister(1234) # for reproducibility
    u = similar(mesh.le_de)
    h = randn(rng, eltype(mesh.Ai), length(mesh.Ai))
    grad! = Operators.Gradient(mesh)
    div! = Operators.Divergence(mesh)
    for i = 1:20
        hmax = normL2(h)
        @. h = inv(hmax) * h
        grad!(u, mgr, h)     # covariant
        @. u *= mesh.le_de   # contravariant
        div!(h, mgr, u)      # density 
        @. h *= mesh.inv_Ai  # scalar
    end
    return inv(sqrt(normL2(h)))::eltype(mesh.le_de)
end
