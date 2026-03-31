#========================== Hyperviscous filter ===========================#

scratch_hyperdiff(sphere::VoronoiSphere, ::Val{:vector_curl}, u) =
    (zv = similar(u, size(sphere.Av)), gradzv = similar(u, size(sphere.lon_e)))

function hyperdiff!(
    ucov_out,
    ucov,
    dissip::HyperDiffusion{:vector_curl},
    sphere::VoronoiSphere,
    dt,
    scratch,
    mgr,
)
    nudt = dissip.nu * dt
    zv =
        voronoi_curl_2D!(mgr, scratch.zv, ucov, sphere.Av, sphere.dual_edge, sphere.dual_ne)
    gradzv = voronoi_gradv_2D!(mgr, scratch.gradzv, zv, sphere.edge_down_up, sphere.le_de)
    for i = 2:dissip.niter
        zv = voronoi_curl_2D!(
            mgr,
            scratch.zv,
            gradzv,
            sphere.Av,
            sphere.dual_edge,
            sphere.dual_ne,
        )
        gradzv =
            voronoi_gradv_2D!(mgr, scratch.gradzv, zv, sphere.edge_down_up, sphere.le_de)
    end
    @. ucov_out = ucov - nudt * gradzv
end

voronoi_curl_2D!(mgr, ::Void, ucov, areas, edges, signs) =
    voronoi_curl_2D!(mgr, similar(areas, eltype(ucov)), ucov, areas, edges, signs)

function voronoi_curl_2D!(mgr, zv, ucov, areas, edges, signs)
    voronoi_curl_2D_(mgr, zv, ucov, areas, edges, signs)
    return zv
end

@loops function voronoi_curl_2D_(_, zv, ucov, areas, edges, signs)
    let ijrange = axes(zv, 1)
        F = eltype(zv)
        @unroll for ij in ijrange
            aa = inv(areas[ij])
            ee = (edges[e, ij] for e = 1:3) # 3 edge indices
            ss = (F(signs[e, ij]) for e = 1:3) # 3 signs
            zv[ij] = aa * sum(ucov[ee[edge]] * ss[edge] for edge = 1:3)
        end
    end
end

voronoi_gradv_2D!(mgr, ::Void, zv, down_up, le_de) =
    voronoi_gradv_2D!(mgr, similar(le_de, eltype(zv)), zv, down_up, le_de)

function voronoi_gradv_2D!(mgr, grad, zv, down_up, le_de)
    voronoi_gradv_2D_(mgr, grad, zv, down_up, le_de)
    return grad
end

@loops function voronoi_gradv_2D_(_, grad, zv, down_up, le_de)
    let ijrange = axes(grad, 1) # velocity points
        for ij in ijrange
            ij_down, ij_up = down_up[1, ij], down_up[2, ij]
            de_le = inv(le_de[ij]) # covariant => contravariant
            grad[ij] = de_le * (zv[ij_up] - zv[ij_down])
        end
    end
end
