@inline loop_cell(output::AbstractArray, args...) = loop_cell(rank(output), output, args...)

@inline function loop_cell(::Val{1}, output, mgr, action!, op, stencil, inputs...)
    @inb for cell in eachindex(output)
        deg = op.primal_deg[cell]
        @unroll deg in 5:7 begin
            st = stencil(op, cell, Val(deg))
            action!(output, st(inputs...), cell)
        end
    end
    return nothing
end

@inline function loop_cell(::Val{2}, output, mgr, action!, op, stencil, inputs...)
    @with mgr, 
    let (krange, irange) = axes(output)
        @inb for cell in irange
            deg = op.primal_deg[cell]
            @unroll deg in 5:7 begin
                st = stencil(op, cell, Val(deg))
                @vec for k in krange
                    action!(output, st(inputs..., k), k, cell)
                end
            end
        end
    end
    return nothing
end

@inline function loop_cell(::Val{3}, output, mgr, action!, op, stencil, inputs...)
    nl = Int32(size(output,1)*size(output,2)) # merged axis
    @with mgr, 
    let (lrange, irange) = (1:nl, axes(output, 3))
        @inb for cell in irange
            deg = op.primal_deg[cell]
            @unroll deg in 5:7 begin
                st = stencil(op, cell, Val(deg))
                @vec for l in lrange
                    k = MergedIndex(l, nl)
                    action!(output, st(inputs..., k), k, cell)
                end
            end
        end
    end
    return nothing
end

@inline loop_trisk(output::AbstractArray, args...) = loop_trisk(rank(output), output, args...)

@inline function loop_trisk(::Val{1}, output, mgr, action!, op, stencil, inputs...)
    @with mgr,
    let irange = eachindex(output)
        @inb for edge in irange
            deg = op.trisk_deg[edge]
            @unroll deg in 9:11 begin
                st = stencil(op, edge, Val(deg))
                action!(output, st(inputs...), edge)
            end
        end
    end
    return nothing
end

@inline function loop_trisk(::Val{2}, output, mgr, action!, op, stencil, inputs...)
    @with mgr, 
    let (krange, irange) = axes(output)
        @inb for edge in irange
            deg = op.trisk_deg[edge]
            @unroll deg in 9:11 begin
                st = stencil(op, edge, Val(deg))
                @vec for k in krange
                    action!(output, st(inputs..., k), k, edge)
                end
            end
        end
    end
    return nothing
end

@inline function loop_trisk(::Val{3}, output, mgr, action!, op, stencil, inputs...)
    nl = Int32(size(output,1)*size(output,2)) # merged axis
    @with mgr, 
    let (lrange, irange) = (1:nl, axes(output, 3))
        @inb for edge in irange
            deg = op.trisk_deg[edge]
            @unroll deg in 9:11 begin
                st = stencil(op, edge, Val(deg))
                @vec for l in lrange
                    k = MergedIndex(l, nl)
                    action!(output, st(inputs..., k), k, edge)
                end
            end
        end
    end
    return nothing
end
