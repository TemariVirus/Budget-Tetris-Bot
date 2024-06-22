const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const Order = std.math.Order;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const Facing = engine.pieces.Facing;
const KickFn = engine.kicks.KickFn;
const Position = engine.pieces.Position;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Rotation = engine.kicks.Rotation;

const root = @import("root.zig");
const Placement = root.Placement;

// TODO: Add pathfinding functions

/// A set of combinations of pieces and their positions, within certain bounds
/// as defined by `shape`.
pub const PiecePosSet = struct {
    const width: usize = 10;
    const height: usize = 24;
    const depth: usize = 4;
    const len = width * height * depth;
    const BackingSet = std.StaticBitSet(len);
    const Self = @This();

    data: BackingSet,

    pub const Iterator = struct {
        set: BackingSet.Iterator(.{}),
        piece: PieceKind,

        pub fn next(self: *Iterator) ?Placement {
            if (self.set.next()) |index| {
                return reverseIndex(self.piece, index);
            }
            return null;
        }
    };

    /// Initialises an empty set.
    pub fn init() Self {
        return Self{
            .data = BackingSet.initEmpty(),
        };
    }

    /// Converts a piece and position to an index into the backing bit set.
    pub fn flatIndex(piece: Piece, pos: Position) usize {
        const facing = @intFromEnum(piece.facing);
        const x: usize = @intCast(pos.x - piece.minX());
        const y: usize = @intCast(pos.y - piece.minY());

        assert(x < width);
        assert(y < height);
        assert(facing < depth);

        return x + y * width + facing * width * height;
    }

    /// Converts an index into the backing bit set to it's coressponding piece and
    /// position.
    pub fn reverseIndex(piece_kind: PieceKind, index: usize) Placement {
        const x = index % width;
        const y = (index / width) % height;
        const facing = index / (width * height);

        const piece = Piece{ .kind = piece_kind, .facing = @enumFromInt(facing) };
        return .{
            .piece = piece,
            .pos = .{
                .x = @as(i8, @intCast(x)) + piece.minX(),
                .y = @as(i8, @intCast(y)) + piece.minY(),
            },
        };
    }

    /// Returns `true` if the set contains the given piece-position combination;
    /// Otherwise, `false`.
    pub fn contains(self: Self, piece: Piece, pos: Position) bool {
        const index = Self.flatIndex(piece, pos);
        return self.data.isSet(index);
    }

    /// Adds the given piece-position combination to the set.
    pub fn put(self: *Self, piece: Piece, pos: Position) void {
        const index = Self.flatIndex(piece, pos);
        self.data.set(index);
    }

    /// Adds the given piece-position combination to the set. Returns `true` if the
    /// combination was already in the set; Otherwise, `false`.
    pub fn putGet(self: *Self, piece: Piece, pos: Position) bool {
        const index = Self.flatIndex(piece, pos);

        const was_set = self.data.isSet(index);
        self.data.set(index);
        return was_set;
    }

    /// Returns an iterator over the set.
    pub fn iterator(self: *const Self, piece_kind: PieceKind) Iterator {
        return Iterator{
            .set = self.data.iterator(.{}),
            .piece = piece_kind,
        };
    }
};

// 200 cells * 4 rotations = 800 intermediate placements at most
const PlacementStack = std.BoundedArray(PiecePosition, 800);
const PiecePosition = packed struct {
    y: i8,
    x: i6,
    facing: Facing,

    pub fn pack(piece: Piece, pos: Position) PiecePosition {
        return PiecePosition{
            .y = pos.y,
            .x = @intCast(pos.x),
            .facing = piece.facing,
        };
    }

    pub fn unpack(self: PiecePosition, piece_kind: PieceKind) Placement {
        return .{
            .piece = .{ .kind = piece_kind, .facing = self.facing },
            .pos = .{ .y = self.y, .x = self.x },
        };
    }
};

const Move = enum {
    left,
    right,
    rotate_cw,
    rotate_double,
    rotate_ccw,
    drop,

    const moves = blk: {
        var m = [_]Move{undefined} ** @typeInfo(Move).Enum.fields.len;
        for (@typeInfo(Move).Enum.fields, 0..) |field, i| {
            m[i] = @enumFromInt(field.value);
        }
        break :blk m;
    };
};

/// Intermediate game state when searching for possible placements.
const Intermediate = struct {
    playfield: BoardMask,
    current: Piece,
    pos: Position,
    kicks: *const KickFn,

    /// Returns `true` if the move was successful. Otherwise, `false`.
    pub fn makeMove(self: *Intermediate, move: Move) bool {
        return switch (move) {
            .left => blk: {
                if (self.pos.x <= self.current.minX()) {
                    break :blk false;
                }
                break :blk self.tryTranspose(.{ .x = self.pos.x - 1, .y = self.pos.y });
            },
            .right => blk: {
                if (self.pos.x >= self.current.maxX()) {
                    break :blk false;
                }
                break :blk self.tryTranspose(.{ .x = self.pos.x + 1, .y = self.pos.y });
            },
            .rotate_cw => self.tryRotate(.quarter_cw),
            .rotate_double => self.tryRotate(.half),
            .rotate_ccw => self.tryRotate(.quarter_ccw),
            .drop => blk: {
                if (self.pos.y <= self.current.minY()) {
                    break :blk false;
                }
                break :blk self.tryTranspose(.{ .x = self.pos.x, .y = self.pos.y - 1 });
            },
        };
    }

    /// Returns `true` if the piece was successfully transposed. Otherwise, `false`.
    fn tryTranspose(self: *Intermediate, pos: Position) bool {
        if (self.playfield.collides(self.current.mask(), pos)) {
            return false;
        }
        self.pos = pos;
        return true;
    }

    /// Returns `true` if the piece was successfully rotated. Otherwise, `false`.
    fn tryRotate(self: *Intermediate, rotation: Rotation) bool {
        const new_piece = Piece{
            .facing = self.current.facing.rotate(rotation),
            .kind = self.current.kind,
        };

        for (self.kicks(self.current, rotation)) |kick| {
            const kicked_pos = self.pos.add(kick);
            if (!self.playfield.collides(new_piece.mask(), kicked_pos)) {
                self.current = new_piece;
                self.pos = kicked_pos;
                return true;
            }
        }

        return false;
    }

    /// Returns `true` if the piece is on the ground. Otherwise, `false`.
    pub fn onGround(self: Intermediate) bool {
        return self.playfield.collides(
            self.current.mask(),
            Position{ .x = self.pos.x, .y = self.pos.y - 1 },
        );
    }
};

/// Returns the set of all placements where the top of the piece does not
/// exceed `max_height`. Assumes that no cells in the playfield are higher than
/// `max_height`.
pub fn allPlacements(
    playfield: BoardMask,
    kicks: *const KickFn,
    piece_kind: PieceKind,
    max_height: u6,
) PiecePosSet {
    var seen = PiecePosSet.init();
    var placements = PiecePosSet.init();
    var stack = PlacementStack.init(0) catch unreachable;

    // Start right above `max_height`
    inline for (@typeInfo(Facing).Enum.fields) |facing| {
        const piece = Piece{
            .facing = @enumFromInt(facing.value),
            .kind = piece_kind,
        };
        const pos = Position{
            .x = 0,
            .y = @as(i8, max_height) + piece.minY(),
        };
        stack.append(PiecePosition.pack(piece, pos)) catch unreachable;
    }

    while (stack.popOrNull()) |placement| {
        const piece, const pos = blk: {
            const temp = placement.unpack(piece_kind);
            break :blk .{ temp.piece, temp.pos };
        };
        if (seen.putGet(piece, pos)) {
            continue;
        }

        for (Move.moves) |move| {
            var new_game = Intermediate{
                .playfield = playfield,
                .current = piece,
                .pos = pos,
                .kicks = kicks,
            };

            // Skip if piece was unable to move
            if (!new_game.makeMove(move)) {
                continue;
            }
            if (seen.contains(new_game.current, new_game.pos)) {
                continue;
            }

            // Branch out after movement if the piece is not too high
            if (new_game.pos.y > @as(i8, max_height) + new_game.current.minY()) {
                continue;
            }
            stack.append(
                PiecePosition.pack(new_game.current, new_game.pos),
            ) catch unreachable;

            // Skip this placement if the piece is too high, or if it's not on the ground
            if (new_game.pos.y + @as(i8, new_game.current.top()) > max_height or
                !new_game.onGround())
            {
                continue;
            }
            placements.put(new_game.current, new_game.pos);
        }
    }

    return placements;
}

pub const MoveNode = struct {
    placement: Placement,
    score: f32,
};
const compareFn = (struct {
    fn cmp(_: void, a: MoveNode, b: MoveNode) Order {
        return std.math.order(b.score, a.score);
    }
}).cmp;
pub const MoveQueue = std.PriorityQueue(MoveNode, void, compareFn);

/// Scores and orders the moves in `moves` based on the `scoreFn`, removing
/// placements where `validFn` returns `false`. Higher scores are dequeued first.
pub fn orderMoves(
    queue: *MoveQueue,
    playfield: BoardMask,
    piece: PieceKind,
    moves: PiecePosSet,
    max_height: u6,
    comptime validFn: fn ([]const u16) bool,
    nn: anytype,
    comptime scoreFn: fn (BoardMask, @TypeOf(nn)) f32,
) void {
    var iter = moves.iterator(piece);
    while (iter.next()) |placement| {
        var board = playfield;
        board.place(placement.piece.mask(), placement.pos);
        const cleared = board.clearLines(placement.pos.y);
        const new_height = max_height - cleared;
        if (!validFn(board.rows[0..new_height])) {
            continue;
        }

        queue.add(.{
            .placement = placement,
            .score = scoreFn(board, nn),
        }) catch unreachable;
    }
}

test allPlacements {
    var playfield = BoardMask{};
    playfield.rows[3] |= 0b0111111110_0;
    playfield.rows[2] |= 0b0010000000_0;
    playfield.rows[1] |= 0b0000001000_0;
    playfield.rows[0] |= 0b0000000001_0;

    const PIECE = PieceKind.l;
    const placements = allPlacements(playfield, engine.kicks.srs, PIECE, 5);

    var iter = placements.iterator(PIECE);
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try expect(count == 25);
}
