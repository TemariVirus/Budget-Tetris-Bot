pub const movegen = @import("movegen.zig");
pub const neat = @import("neat.zig");
pub const pc = @import("pc.zig");

const std = @import("std");

const engine = @import("engine");
const Piece = engine.pieces.Piece;
const Position = engine.pieces.Position;

pub const Placement = struct {
    piece: Piece,
    pos: Position,
};

test {
    std.testing.refAllDecls(@This());
}
